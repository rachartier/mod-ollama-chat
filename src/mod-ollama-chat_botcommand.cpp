#include "mod-ollama-chat_botcommand.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_handler.h"

#include "Player.h"
#include "Unit.h"
#include "PlayerbotAI.h"
#include "PlayerbotMgr.h"
#include "ObjectAccessor.h"
#include "ObjectMgr.h"
#include "Item.h"
#include "ItemTemplate.h"
#include "Bag.h"
#include "SpellMgr.h"
#include "SpellInfo.h"
#include "SharedDefines.h"
#include "MotionMaster.h"
#include "Log.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <utility>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    enum class BotIntent { None, Heal, Give, Buff, Come, Follow, AskHave, Help, ListInventory };

    struct GiveRequest
    {
        std::string item;          // raw item phrase, resolved against the bot's bags
        uint32      quantity = 1;
        bool        all      = false;
    };

    struct ParsedCommand
    {
        BotIntent                intent = BotIntent::None;
        std::string              itemName;     // for AskHave
        std::vector<GiveRequest> gives;        // for Give (one or more items)
    };

    // Work that touches game state is queued here from the async LLM thread and run
    // on the world thread by PumpBotCommandTasks (OnUpdate), so bot actions never
    // mutate the world off-thread.
    std::queue<std::function<void()>> g_taskQueue;
    std::mutex g_taskMutex;
    void EnqueueTask(std::function<void()> t)
    {
        std::lock_guard<std::mutex> lock(g_taskMutex);
        g_taskQueue.push(std::move(t));
    }

    std::string ToLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return s;
    }

    std::string Trim(const std::string& s)
    {
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        return s.substr(a, b - a + 1);
    }

    PlayerbotAI* BotAI(Player* p)
    {
        return p ? PlayerbotsMgr::instance().GetPlayerbotAI(p) : nullptr;
    }

    bool IsBot(Player* p)
    {
        PlayerbotAI* ai = BotAI(p);
        return ai && ai->IsBotAI();
    }

    // All commandable bots within range on the player's map, nearest first.
    std::vector<Player*> GatherCommandableBots(Player* player)
    {
        std::vector<std::pair<float, Player*>> found;
        for (auto const& pair : ObjectAccessor::GetPlayers())
        {
            Player* candidate = pair.second;
            if (!candidate || candidate == player) continue;
            if (!candidate->IsInWorld() || !candidate->IsAlive()) continue;
            if (candidate->GetMap() != player->GetMap()) continue;
            if (!IsBot(candidate)) continue;
            float d = candidate->GetDistance(player);
            if (d <= g_BotCommandRange) found.push_back({ d, candidate });
        }
        std::sort(found.begin(), found.end(),
                  [](auto const& a, auto const& b) { return a.first < b.first; });
        std::vector<Player*> bots;
        for (auto const& f : found) bots.push_back(f.second);
        return bots;
    }

    template <class F>
    void ForEachBotBagItem(Player* bot, F f)
    {
        for (uint8 slot = INVENTORY_SLOT_ITEM_START; slot < INVENTORY_SLOT_ITEM_END; ++slot)
            if (Item* it = bot->GetItemByPos(INVENTORY_SLOT_BAG_0, slot))
                f(it);
        for (uint8 bag = INVENTORY_SLOT_BAG_START; bag < INVENTORY_SLOT_BAG_END; ++bag)
            if (Bag* b = bot->GetBagByPos(bag))
                for (uint32 i = 0; i < b->GetBagSize(); ++i)
                    if (Item* it = b->GetItemByPos(i))
                        f(it);
    }



    // Tokenize a name query into lowercase, de-pluralized words, dropping command
    // and filler words. Lets "give me all your mana potions" match "Minor Mana Potion".
    std::vector<std::string> ItemQueryTokens(const std::string& phrase)
    {
        static const std::vector<std::string> stop = {
            "give", "gimme", "hand", "over", "share", "trade", "me", "us", "all",
            "your", "you", "do", "have", "has", "had", "got", "any", "some", "a",
            "an", "the", "of", "please", "can", "could", "would", "will", "to",
            "for", "need", "want", "i", "my", "everything"
        };
        std::vector<std::string> out;
        std::istringstream iss(ToLower(phrase));
        std::string word;
        while (iss >> word)
        {
            std::string w;
            for (char c : word)
                if (std::isalnum((unsigned char)c)) w += c;
            if (w.empty()) continue;
            if (std::all_of(w.begin(), w.end(), [](unsigned char c) { return std::isdigit(c); }))
                continue; // quantities like "5" are not part of the item name
            if (std::find(stop.begin(), stop.end(), w) != stop.end()) continue;
            if (w.size() > 3 && w.back() == 's') w.pop_back(); // crude singularize
            out.push_back(w);
        }
        return out;
    }

    // Extracts item entry ids from any WoW item links (shift-clicked items) in the
    // text, e.g. "...|Hitem:6149:0:...|h[Minor Mana Potion]|h...". Case-insensitive.
    std::vector<uint32> ExtractItemLinkEntries(const std::string& s)
    {
        std::string l = ToLower(s);
        std::vector<uint32> out;
        size_t pos = 0;
        while ((pos = l.find("hitem:", pos)) != std::string::npos)
        {
            pos += 6;
            uint32 id = 0;
            while (pos < l.size() && std::isdigit((unsigned char)l[pos])) { id = id * 10 + (l[pos] - '0'); ++pos; }
            if (id) out.push_back(id);
        }
        return out;
    }

    uint32 FirstItemLinkEntry(const std::string& s)
    {
        std::vector<uint32> e = ExtractItemLinkEntries(s);
        return e.empty() ? 0 : e.front();
    }

    // Resolve a name query against the items in the bot's bags. Returns the entry
    // of the best (highest-count) match whose name contains every query token, or
    // 0 if nothing matches. Fills outName/outCount for the chosen entry.
    uint32 ResolveBotItem(Player* bot, const std::string& query, std::string& outName, uint32& outCount)
    {
        std::vector<std::string> tokens = ItemQueryTokens(query);
        if (tokens.empty()) return 0;

        std::map<uint32, std::pair<std::string, uint32>> inv; // entry -> (name, total count)
        ForEachBotBagItem(bot, [&](Item* it) {
            ItemTemplate const* t = it->GetTemplate();
            if (!t) return;
            auto& e = inv[it->GetEntry()];
            e.first = t->Name1;
            e.second += it->GetCount();
        });

        uint32 best = 0, bestCount = 0;
        for (auto const& kv : inv)
        {
            std::string n = ToLower(kv.second.first);
            bool all = true;
            for (auto const& tok : tokens)
                if (n.find(tok) == std::string::npos) { all = false; break; }
            if (all && kv.second.second > bestCount)
            {
                bestCount = kv.second.second;
                best      = kv.first;
                outName   = kv.second.first;
                outCount  = kv.second.second;
            }
        }
        return best;
    }


    // Sends an in-character reply to the player. Uses the LLM (bot personality)
    // when enabled, otherwise the plain fallback line. The LLM call and the
    // resulting whisper run on a detached thread, matching the module's existing
    // async chat pattern.
    void Respond(Player* bot, Player* player, const std::string& situation, const std::string& fallback)
    {
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return;

        if (!g_BotCommandLLMReplies)
        {
            ai->Whisper(fallback, player->GetName());
            return;
        }

        std::string persona = GetPersonalityPromptAddition(GetBotPersonality(bot));
        std::string prompt =
            "You are " + std::string(bot->GetName()) + ", a World of Warcraft companion. " + persona +
            " You are speaking to " + std::string(player->GetName()) + ". " + situation +
            " Reply with ONE short, natural, in-character sentence. No asterisks, no emotes, no quotes.";

        ObjectGuid botGuid = bot->GetGUID();
        ObjectGuid playerGuid = player->GetGUID();
        std::future<std::string> fut = SubmitQuery(prompt);
        std::thread([botGuid, playerGuid, fallback, fut = std::move(fut)]() mutable {
            std::string resp;
            try { resp = fut.get(); } catch (...) { resp = ""; }
            if (!IsValidAPIResponse(resp)) resp = "";

            Player* b = ObjectAccessor::FindPlayer(botGuid);
            Player* p = ObjectAccessor::FindPlayer(playerGuid);
            if (!b || !p) return;
            PlayerbotAI* bai = PlayerbotsMgr::instance().GetPlayerbotAI(b);
            if (!bai) return;
            bai->Whisper(resp.empty() ? fallback : resp, p->GetName());
        }).detach();
    }

    // --- Action handlers ---------------------------------------------------

    // Best castable healing spell the bot can cast on the player (0 = none).
    uint32 FindHealSpell(Player* bot, Player* player)
    {
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return 0;
        uint32 bestSpell = 0, bestLevel = 0;
        for (auto const& sp : bot->GetSpellMap())
        {
            uint32 spellId = sp.first;
            SpellInfo const* info = sSpellMgr->GetSpellInfo(spellId);
            if (!info) continue;
            if (info->Attributes & SPELL_ATTR0_PASSIVE) continue;
            if (bot->HasSpellCooldown(spellId)) continue;
            bool isHeal = false;
            for (int i = 0; i < MAX_SPELL_EFFECTS; ++i)
                if (info->Effects[i].IsEffect() && info->Effects[i].Effect == SPELL_EFFECT_HEAL)
                    isHeal = true;
            if (!isHeal) continue;
            if (!ai->CanCastSpell(spellId, player)) continue;
            if (info->SpellLevel >= bestLevel) { bestLevel = info->SpellLevel; bestSpell = spellId; }
        }
        return bestSpell;
    }

    // First castable beneficial aura the bot can cast on the player (0 = none).
    uint32 FindBuffSpell(Player* bot, Player* player)
    {
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return 0;
        for (auto const& sp : bot->GetSpellMap())
        {
            uint32 spellId = sp.first;
            SpellInfo const* info = sSpellMgr->GetSpellInfo(spellId);
            if (!info) continue;
            if (info->Attributes & SPELL_ATTR0_PASSIVE) continue;
            if (!info->IsPositive()) continue;
            if (bot->HasSpellCooldown(spellId)) continue;
            bool isAura = false;
            for (int i = 0; i < MAX_SPELL_EFFECTS; ++i)
                if (info->Effects[i].IsEffect() && info->Effects[i].Effect == SPELL_EFFECT_APPLY_AURA)
                    isAura = true;
            if (!isAura) continue;
            if (!ai->CanCastSpell(spellId, player)) continue;
            return spellId;
        }
        return 0;
    }

    void DoHeal(Player* bot, Player* player)
    {
        if (!g_BotCommandAllowHeal) { Respond(bot, player, "You cannot heal.", "I can't heal."); return; }
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return;
        uint32 spell = FindHealSpell(bot, player);
        if (!spell)
        {
            Respond(bot, player, "The player asked for a heal but you have no healing spell you can cast on them.",
                    "Sorry, I can't heal you.");
            return;
        }
        ai->CastSpell(spell, player);
        Respond(bot, player, "You just cast a healing spell on the player who asked to be healed.",
                "Healing you now!");
    }

    void DoBuff(Player* bot, Player* player)
    {
        if (!g_BotCommandAllowBuff) { Respond(bot, player, "You cannot buff.", "I can't buff you."); return; }
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return;
        // ponytail: naive "first castable positive aura" pick; refine the filter if it picks junk.
        uint32 spell = FindBuffSpell(bot, player);
        if (!spell)
        {
            Respond(bot, player, "The player asked for a buff but you have none to give them.",
                    "I don't have a buff for you.");
            return;
        }
        ai->CastSpell(spell, player);
        Respond(bot, player, "You just cast a beneficial buff on the player who asked for one.",
                "There, you're buffed!");
    }

    void DoMove(Player* bot, Player* player, bool follow)
    {
        if (!g_BotCommandAllowMove) return;
        // ponytail: raw MotionMaster command; the playerbots AI tick can override
        // it on its next decision. Hook a playerbots movement strategy if it must stick.
        if (follow)
        {
            bot->GetMotionMaster()->MoveFollow(player, 1.5f, bot->GetFollowAngle());
            Respond(bot, player, "You are now following the player as they asked.", "Right behind you.");
        }
        else
        {
            bot->GetMotionMaster()->MovePoint(0, player->GetPositionX(),
                                              player->GetPositionY(), player->GetPositionZ());
            Respond(bot, player, "You are walking over to the player who called you.", "On my way.");
        }
    }

    // "help me!" - assist the player: attack what they're fighting and close in,
    // otherwise just move to them.
    // ponytail: movement/attack are nudged via MotionMaster; the playerbots AI tick
    // can override them. Reliable assist needs the bot grouped with the player or a
    // deeper playerbots-strategy integration.
    void DoHelp(Player* bot, Player* player)
    {
        if (!g_BotCommandAllowMove) return;

        Unit* target = player->GetVictim();
        if (!target) target = player->GetSelectedUnit();

        if (target && target->IsAlive() && bot->IsValidAttackTarget(target))
        {
            bot->Attack(target, true);
            bot->GetMotionMaster()->MoveChase(target);
            Respond(bot, player, "The player called for help while fighting. You are rushing in to attack their enemy.",
                    "I've got your back!");
        }
        else
        {
            bot->GetMotionMaster()->MovePoint(0, player->GetPositionX(),
                                              player->GetPositionY(), player->GetPositionZ());
            Respond(bot, player, "The player called for help. You are heading over to them.", "Coming to help!");
        }
    }

    void DoAskHave(Player* bot, Player* player, const std::string& query)
    {
        std::string name;
        uint32 count = 0;
        uint32 entry = 0;
        if (uint32 linkEntry = FirstItemLinkEntry(query))
        {
            ItemTemplate const* t = sObjectMgr->GetItemTemplate(linkEntry);
            name  = t ? t->Name1 : "that";
            count = bot->GetItemCount(linkEntry, false);
            entry = count > 0 ? linkEntry : 0;
        }
        else
        {
            entry = ResolveBotItem(bot, query, name, count);
        }

        if (!entry)
        {
            std::string display = name.empty() ? query : name;
            Respond(bot, player,
                    "The player asked whether you have any '" + display + "'. You have none in your bags. Tell them you don't have any.",
                    "No, I don't have any " + display + ".");
            return;
        }
        Respond(bot, player,
                "The player asked whether you have any '" + query + "'. You have " + std::to_string(count) + "x " + name +
                    " in your bags. Tell them yes and that they can have some.",
                "Yes, I have " + std::to_string(count) + "x " + name + ".");
    }

    // Lists the bot's bag contents to the player, factually (not via the LLM, so
    // nothing is dropped), chunked to fit whisper length limits.
    void DoListInventory(Player* bot, Player* player)
    {
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return;

        std::map<std::string, uint32> items; // name -> total count, sorted by name
        ForEachBotBagItem(bot, [&](Item* it) {
            if (ItemTemplate const* t = it->GetTemplate()) items[t->Name1] += it->GetCount();
        });

        if (items.empty())
        {
            ai->Whisper("My bags are empty.", player->GetName());
            return;
        }

        std::string line = "I have: ";
        bool first = true;
        for (auto const& kv : items)
        {
            std::string piece = (first ? "" : ", ") + std::to_string(kv.second) + "x " + kv.first;
            if (line.size() + piece.size() > 230)
            {
                ai->Whisper(line, player->GetName());
                line = "";
                piece = std::to_string(kv.second) + "x " + kv.first;
            }
            line += piece;
            first = false;
        }
        if (!line.empty()) ai->Whisper(line, player->GetName());
    }

    std::string JoinComma(const std::vector<std::string>& parts)
    {
        std::string out;
        for (size_t i = 0; i < parts.size(); ++i)
            out += (i ? ", " : "") + parts[i];
        return out;
    }

    // Transfers up to the requested amount of a known item directly into the
    // player's bags. Never creates items: the bot loses exactly what the player
    // gains. Returns the number actually given.
    uint32 TransferItem(Player* bot, Player* player, uint32 entry, uint32 quantity, bool all)
    {
        uint32 have = bot->GetItemCount(entry, false);
        if (have == 0) return 0;

        uint32 want = all ? have : std::min(quantity, have);
        if (g_BotCommandMaxGiveQuantity > 0)
            want = std::min(want, g_BotCommandMaxGiveQuantity);
        if (want == 0) return 0;

        ItemPosCountVec dest;
        uint32 noSpace = 0;
        player->CanStoreNewItem(NULL_BAG, NULL_SLOT, dest, entry, want, &noSpace);
        uint32 give = want - noSpace;
        if (give == 0) return 0;

        bot->DestroyItemCount(entry, give, true);
        Item* stored = player->StoreNewItem(dest, entry, true, Item::GenerateItemRandomPropertyId(entry));
        if (stored) player->SendNewItem(stored, give, true, false);
        return give;
    }

    // Resolves a give request to a bot inventory entry: a shift-clicked item link
    // first (exact id), else the LLM-provided item name matched against the bags.
    uint32 ResolveGiveEntry(Player* bot, const GiveRequest& req, std::string& outName)
    {
        if (uint32 linkEntry = FirstItemLinkEntry(req.item))
        {
            ItemTemplate const* t = sObjectMgr->GetItemTemplate(linkEntry);
            outName = t ? t->Name1 : "that item";
            return bot->GetItemCount(linkEntry, false) > 0 ? linkEntry : 0;
        }
        uint32 have = 0;
        return ResolveBotItem(bot, req.item, outName, have);
    }

    // Gives one or more items straight into the player's bags (direct transfer, no
    // trade window). Honours exact quantities, multiple items and shift-clicked item
    // links. The bot only ever moves items it actually owns.
    void DirectGive(Player* bot, Player* player, const std::vector<GiveRequest>& gives)
    {
        if (!g_BotCommandAllowGive) { Respond(bot, player, "You cannot give items.", "I can't give you anything right now."); return; }

        std::vector<std::string> gave;
        std::vector<std::string> missing;
        for (GiveRequest const& req : gives)
        {
            std::string name;
            uint32 entry = ResolveGiveEntry(bot, req, name);
            if (!entry) { missing.push_back(name.empty() ? req.item : name); continue; }

            uint32 given = TransferItem(bot, player, entry, req.quantity, req.all);
            if (given > 0) gave.push_back(std::to_string(given) + "x " + name);
            else           missing.push_back(name);
        }

        if (gave.empty())
        {
            Respond(bot, player,
                    "The player asked you for " + JoinComma(missing) + " but you have none of it (or their bags are full).",
                    "Sorry, I don't have any " + JoinComma(missing) + ".");
            return;
        }

        std::string situation = "You just handed the player " + JoinComma(gave) + ".";
        std::string fallback  = "Here you go: " + JoinComma(gave) + ".";
        if (!missing.empty())
        {
            situation += " You had none of: " + JoinComma(missing) + ".";
            fallback  += " (I didn't have any " + JoinComma(missing) + ".)";
        }
        Respond(bot, player, situation, fallback);
    }

    void ExecuteParsed(Player* player, Player* bot, const ParsedCommand& cmd)
    {
        if (!player || !bot) return;
        if (bot->GetMap() != player->GetMap()) return;
        if (bot->GetDistance(player) > g_BotCommandRange) return;

        switch (cmd.intent)
        {
            case BotIntent::Heal:          DoHeal(bot, player); break;
            case BotIntent::Buff:          DoBuff(bot, player); break;
            case BotIntent::Give:          DirectGive(bot, player, cmd.gives); break;
            case BotIntent::Come:          DoMove(bot, player, false); break;
            case BotIntent::Follow:        DoMove(bot, player, true); break;
            case BotIntent::Help:          DoHelp(bot, player); break;
            case BotIntent::AskHave:       DoAskHave(bot, player, cmd.itemName); break;
            case BotIntent::ListInventory: DoListInventory(bot, player); break;
            default: break;
        }
    }

    // --- Intent pipeline ---------------------------------------------------

    // Synchronous guard: does the message look like a "give" request? Only used to
    // suppress it from playerbots (so it doesn't open its own trade window). The real
    // intent is decided by the LLM.
    bool LooksLikeGive(const std::string& msgLower)
    {
        static const char* cues[] = { "give", "gimme", "trade", "hand over", "share", "hitem:" };
        for (const char* c : cues)
            if (msgLower.find(c) != std::string::npos) return true;
        return false;
    }

    // Coarse gate for /say only (whispers always run the LLM): avoids an LLM call on
    // every nearby line. A directed conversation is what whispers are for.
    bool LooksLikePotentialCommand(const std::string& msgLower)
    {
        static const char* cues[] = {
            "heal", "give", "gimme", "buff", "come", "follow", "help", "assist", "need",
            "want", "cast", "trade", "share", "hand over", "water", "food", "mana",
            "drink", "have", "got", "any", "potion", "?", "inventory", "bag", "carry",
            "hitem:"
        };
        for (const char* c : cues)
            if (msgLower.find(c) != std::string::npos) return true;
        return false;
    }

    // Recent conversation turns (player + bot) for this pair, for the intent prompt.
    std::string RecentHistory(uint64 botGuid, uint64 playerGuid)
    {
        if (!g_EnableChatHistory) return "";
        std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
        auto bi = g_BotConversationHistory.find(botGuid);
        if (bi == g_BotConversationHistory.end()) return "";
        auto pi = bi->second.find(playerGuid);
        if (pi == bi->second.end()) return "";
        std::string out;
        for (auto const& turn : pi->second)
            out += "Player: " + turn.first + "\nBot: " + turn.second + "\n";
        return out;
    }

    // Distinct item names in the bot's bags (capped), so the LLM returns exact names.
    std::string BotItemNames(Player* bot)
    {
        std::vector<std::string> names;
        std::map<std::string, bool> seen;
        ForEachBotBagItem(bot, [&](Item* it) {
            if (ItemTemplate const* t = it->GetTemplate())
                if (!seen.count(t->Name1)) { seen[t->Name1] = true; names.push_back(t->Name1); }
        });
        std::string out;
        for (size_t i = 0; i < names.size() && i < 40; ++i) out += (i ? ", " : "") + names[i];
        return out;
    }

    // Picks the bot that should act on the parsed command among the in-range
    // candidates (nearest first): the nearest one that can actually do it, else the
    // nearest. So "heal me" near five bots is answered by a healer, not a warrior.
    Player* RouteBot(const ParsedCommand& cmd, const std::vector<Player*>& bots, Player* player)
    {
        if (bots.empty()) return nullptr;
        switch (cmd.intent)
        {
            case BotIntent::Heal:
                for (Player* b : bots) if (FindHealSpell(b, player)) return b;
                break;
            case BotIntent::Buff:
                for (Player* b : bots) if (FindBuffSpell(b, player)) return b;
                break;
            case BotIntent::Give:
                for (Player* b : bots)
                    for (GiveRequest const& req : cmd.gives) { std::string n; if (ResolveGiveEntry(b, req, n)) return b; }
                break;
            case BotIntent::AskHave:
                for (Player* b : bots) { std::string n; uint32 c = 0; if (ResolveBotItem(b, cmd.itemName, n, c)) return b; }
                break;
            default: break;
        }
        return bots.front();
    }

    BotIntent IntentFromString(const std::string& s)
    {
        if (s == "heal")    return BotIntent::Heal;
        if (s == "give")    return BotIntent::Give;
        if (s == "buff")    return BotIntent::Buff;
        if (s == "come")    return BotIntent::Come;
        if (s == "follow")  return BotIntent::Follow;
        if (s == "help")    return BotIntent::Help;
        if (s == "askhave") return BotIntent::AskHave;
        if (s == "listinventory") return BotIntent::ListInventory;
        return BotIntent::None;
    }

    std::string CommandSummary(const ParsedCommand& cmd)
    {
        switch (cmd.intent)
        {
            case BotIntent::Heal:          return "Cast a heal on the player.";
            case BotIntent::Buff:          return "Buffed the player.";
            case BotIntent::Come:          return "Walked over to the player.";
            case BotIntent::Follow:        return "Started following the player.";
            case BotIntent::Help:          return "Came to help the player fight.";
            case BotIntent::ListInventory: return "Listed inventory to the player.";
            case BotIntent::AskHave:       return "Answered about " + cmd.itemName + ".";
            case BotIntent::Give:
            {
                std::vector<std::string> names;
                for (GiveRequest const& r : cmd.gives) names.push_back(r.item);
                return "Gave the player: " + JoinComma(names) + ".";
            }
            default: return "";
        }
    }

    // Single LLM call that interprets the message (intent + items) using the
    // conversation history, then routes the action to a capable nearby bot, or, for a
    // whisper with no action, produces a normal chat reply. Replaces all regex parsing.
    void RunIntent(uint64 playerGuid, std::vector<uint64> botGuids, const std::string& msg, bool whisper)
    {
        if (botGuids.empty()) return;

        Player* nearest = ObjectAccessor::FindPlayer(ObjectGuid(botGuids.front()));
        if (!nearest) return;

        std::string history  = RecentHistory(botGuids.front(), playerGuid);
        std::string itemList = BotItemNames(nearest);

        std::string prompt =
            "You interpret a player's message to a World of Warcraft companion bot. "
            "Reply with ONLY compact JSON, no prose, no markdown.\n"
            "Schema: {\"intent\":\"heal|give|buff|come|follow|help|askhave|listinventory|none\","
            "\"items\":[{\"item\":\"<exact item name>\",\"quantity\":<integer>,\"all\":<true|false>}]}\n"
            "Use \"none\" when it is just conversation, not a request to act. "
            "\"help\" = come assist/rescue in a fight. \"askhave\" = asking if you have a specific item. "
            "\"listinventory\" = asking what you carry in general. "
            "For give/askhave, copy item names EXACTLY from what the bot carries: [" + itemList + "]. "
            "Use the conversation to resolve references like \"5 more\". "
            "quantity is a number; all=true means the whole stack.\n"
            "Example: \"hand me a couple bandages\" -> "
            "{\"intent\":\"give\",\"items\":[{\"item\":\"Linen Bandage\",\"quantity\":2,\"all\":false}]}\n"
            + (history.empty() ? "" : ("Recent conversation:\n" + history))
            + "Player says: \"" + msg + "\"";

        // The LLM call + JSON parse run on this detached thread (no game state); the
        // resulting action is queued back to the world thread so nothing mutates
        // off-thread. The intent call needs a bigger token budget than the chat default
        // (g_OllamaNumPredict, ~40) or its JSON gets truncated, and a low temperature
        // for reliable structure - so it calls QueryOllamaAPI directly with overrides.
        // ponytail: this bypasses the query concurrency limit; route through SubmitQuery
        // if command volume ever overloads Ollama.
        std::thread([playerGuid, botGuids, msg, whisper, prompt]() {
            std::string resp;
            try { resp = QueryOllamaAPI(prompt, 1024, 0.1f, /*rawMode*/ true); } catch (...) { return; }
            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[Ollama Chat] Intent raw response: {}", resp);

            ParsedCommand cmd;
            size_t a = resp.find('{');
            size_t b = resp.rfind('}');
            if (a != std::string::npos && b != std::string::npos && b > a)
            {
                try
                {
                    nlohmann::json j = nlohmann::json::parse(resp.substr(a, b - a + 1));
                    cmd.intent = IntentFromString(j.value("intent", "none"));
                    if (j.contains("items") && j["items"].is_array())
                    {
                        for (auto const& it : j["items"])
                        {
                            GiveRequest r;
                            if (it.contains("item") && it["item"].is_string()) r.item = it["item"].get<std::string>();
                            if (it.contains("all") && it["all"].is_boolean())   r.all = it["all"].get<bool>();
                            if (it.contains("quantity") && it["quantity"].is_number())
                                r.quantity = static_cast<uint32>(it["quantity"].get<double>());
                            if (r.quantity == 0) { r.all = true; r.quantity = 1; }
                            if (!r.item.empty()) cmd.gives.push_back(r);
                        }
                    }
                    if (cmd.intent == BotIntent::AskHave && !cmd.gives.empty())
                        cmd.itemName = cmd.gives.front().item;
                }
                catch (...) { cmd.intent = BotIntent::None; }
            }

            if (g_DebugEnabled)
                LOG_INFO("server.loading", "[Ollama Chat] Intent parsed: intent={} items={}",
                         static_cast<int>(cmd.intent), cmd.gives.size());

            // Hand the parsed result back to the world thread for routing + execution,
            // so all game-state access happens on the main thread, never here.
            EnqueueTask([playerGuid, botGuids, msg, whisper, cmd]() {
                Player* player = ObjectAccessor::FindPlayer(ObjectGuid(playerGuid));
                if (!player) return;

                // Re-acquire candidate bots still valid. Proximity candidates must still
                // be on the player's map; a whispered bot stays a candidate regardless
                // (chat works cross-map; actions are distance-guarded in ExecuteParsed).
                std::vector<Player*> bots;
                for (uint64 g : botGuids)
                {
                    Player* b = ObjectAccessor::FindPlayer(ObjectGuid(g));
                    if (!b || !b->IsInWorld()) continue;
                    if (!whisper && b->GetMap() != player->GetMap()) continue;
                    bots.push_back(b);
                }
                if (bots.empty()) return;

                if (cmd.intent == BotIntent::None)
                {
                    if (whisper) OllamaWhisperChatReply(bots.front(), player, msg); // plain chat
                    return;
                }

                Player* target = RouteBot(cmd, bots, player);
                if (!target) return;
                ExecuteParsed(player, target, cmd);
                AppendBotConversation(target->GetGUID().GetRawValue(), playerGuid, msg, CommandSummary(cmd));
            });
        }).detach();
    }

    // Runs queued bot-command work on the world thread (called from OnUpdate).
    void DrainTasks()
    {
        std::queue<std::function<void()>> local;
        {
            std::lock_guard<std::mutex> lock(g_taskMutex);
            std::swap(local, g_taskQueue);
        }
        while (!local.empty())
        {
            if (local.front()) local.front()();
            local.pop();
        }
    }
}

void PumpBotCommandTasks()
{
    DrainTasks();
}

BotCommandResult TryHandleBotCommand(Player* player, const std::string& msg, Player* whisperTarget)
{
    if (!g_BotCommandEnable) return BotCommandResult::NotHandled;
    if (!player || IsBot(player)) return BotCommandResult::NotHandled;

    std::string trimmed = Trim(msg);
    if (trimmed.empty()) return BotCommandResult::NotHandled;
    std::string lower = ToLower(trimmed);

    bool whisperedToBot = whisperTarget && IsBot(whisperTarget);

    // For /say (not a whisper), only engage when the line plausibly is a command, so we
    // don't run the LLM on every nearby remark. Whispers always go to the LLM (that is
    // the conversation channel).
    if (!whisperedToBot && !LooksLikePotentialCommand(lower))
        return BotCommandResult::NotHandled;

    // Candidate bots: the whispered bot, or every commandable bot within range.
    std::vector<Player*> bots;
    if (whisperedToBot)
        bots.push_back(whisperTarget);
    else
        bots = GatherCommandableBots(player);
    if (bots.empty()) return BotCommandResult::NotHandled;

    std::vector<uint64> guids;
    for (Player* b : bots) guids.push_back(b->GetGUID().GetRawValue());

    RunIntent(player->GetGUID().GetRawValue(), guids, trimmed, whisperedToBot);

    // The parse is async, so suppression is decided by a coarse synchronous guard: a
    // give-like message is suppressed so playerbots does not open its own trade window.
    if (LooksLikeGive(lower)) return BotCommandResult::HandledSuppress;

    // Whisper: we own the reply (action or chat) - skip the normal chatter. /say
    // command-like: we acted, skip too.
    return BotCommandResult::Handled;
}
