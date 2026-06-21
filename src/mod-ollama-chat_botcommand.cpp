#include "mod-ollama-chat_botcommand.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_personality.h"

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
#include <map>
#include <mutex>
#include <regex>
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

    // Session-only memory of the last item each player discussed with each bot, so
    // a follow-up like "give me 5 more" resolves to the previous item. In-memory,
    // cleared on restart, never persisted.
    struct GiveMemory { uint32 entry = 0; std::string name; };
    std::map<std::pair<uint64, uint64>, GiveMemory> g_lastGive;
    std::mutex g_lastGiveMutex;

    GiveMemory GetLastGive(Player* player, Player* bot)
    {
        std::lock_guard<std::mutex> lock(g_lastGiveMutex);
        auto it = g_lastGive.find({ player->GetGUID().GetRawValue(), bot->GetGUID().GetRawValue() });
        return it != g_lastGive.end() ? it->second : GiveMemory{};
    }

    void SetLastGive(Player* player, Player* bot, uint32 entry, const std::string& name)
    {
        std::lock_guard<std::mutex> lock(g_lastGiveMutex);
        g_lastGive[{ player->GetGUID().GetRawValue(), bot->GetGUID().GetRawValue() }] = GiveMemory{ entry, name };
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

    Player* FindNearestCommandableBot(Player* player)
    {
        Player* best = nullptr;
        float   bestDist = g_BotCommandRange;

        for (auto const& pair : ObjectAccessor::GetPlayers())
        {
            Player* candidate = pair.second;
            if (!candidate || candidate == player) continue;
            if (!candidate->IsInWorld() || !candidate->IsAlive()) continue;
            if (candidate->GetMap() != player->GetMap()) continue;
            if (!IsBot(candidate)) continue;
            float d = candidate->GetDistance(player);
            if (d <= bestDist)
            {
                bestDist = d;
                best = candidate;
            }
        }
        return best;
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
            "for", "need", "want", "i", "my", "got", "everything",
            // contextual words: when the item phrase is only these, it refers back
            // to the last item discussed (handled via session memory).
            "more", "another", "same", "again", "extra", "additional", "them", "those", "it"
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

    void DoHeal(Player* bot, Player* player)
    {
        if (!g_BotCommandAllowHeal) { Respond(bot, player, "You cannot heal.", "I can't heal."); return; }
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return;

        uint32 bestSpell = 0;
        uint32 bestLevel = 0;
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

            if (info->SpellLevel >= bestLevel)
            {
                bestLevel = info->SpellLevel;
                bestSpell = spellId;
            }
        }

        if (!bestSpell)
        {
            Respond(bot, player, "The player asked for a heal but you have no healing spell you can cast on them.",
                    "Sorry, I can't heal you.");
            return;
        }
        ai->CastSpell(bestSpell, player);
        Respond(bot, player, "You just cast a healing spell on the player who asked to be healed.",
                "Healing you now!");
    }

    void DoBuff(Player* bot, Player* player)
    {
        if (!g_BotCommandAllowBuff) { Respond(bot, player, "You cannot buff.", "I can't buff you."); return; }
        PlayerbotAI* ai = BotAI(bot);
        if (!ai) return;

        // ponytail: naive "first castable positive aura" pick; refine the filter
        // if it ever chooses a junk buff.
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

            ai->CastSpell(spellId, player);
            Respond(bot, player, "You just cast a beneficial buff on the player who asked for one.",
                    "There, you're buffed!");
            return;
        }
        Respond(bot, player, "The player asked for a buff but you have none to give them.",
                "I don't have a buff for you.");
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
        SetLastGive(player, bot, entry, name); // so a follow-up "give me some" refers to it
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
        size_t shown = 0;
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
            ++shown;
        }
        if (!line.empty()) ai->Whisper(line, player->GetName());
        (void)shown;
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

    // Gives one or more items straight into the player's bags (direct transfer, no
    // trade window). Honours exact quantities, supports multiple items, shift-clicked
    // item links, and contextual follow-ups ("5 more") via session memory. The bot
    // only ever moves items it actually owns.
    void DirectGive(Player* bot, Player* player, const std::vector<GiveRequest>& gives)
    {
        if (!g_BotCommandAllowGive) { Respond(bot, player, "You cannot give items.", "I can't give you anything right now."); return; }

        std::vector<std::string> gave;
        std::vector<std::string> missing;
        for (GiveRequest const& req : gives)
        {
            uint32 entry = 0;
            std::string name;
            if (uint32 linkEntry = FirstItemLinkEntry(req.item))
            {
                // shift-clicked item link -> use its id directly
                ItemTemplate const* t = sObjectMgr->GetItemTemplate(linkEntry);
                name  = t ? t->Name1 : "that item";
                if (bot->GetItemCount(linkEntry, false) == 0) { missing.push_back(name); continue; }
                entry = linkEntry;
            }
            else if (ItemQueryTokens(req.item).empty())
            {
                GiveMemory mem = GetLastGive(player, bot); // contextual "5 more"
                entry = mem.entry;
                name  = mem.name;
                if (!entry) { missing.push_back("that"); continue; }
            }
            else
            {
                uint32 have = 0;
                entry = ResolveBotItem(bot, req.item, name, have);
                if (!entry) { missing.push_back(name.empty() ? req.item : name); continue; }
            }

            uint32 given = TransferItem(bot, player, entry, req.quantity, req.all);
            if (given > 0)
            {
                gave.push_back(std::to_string(given) + "x " + name);
                SetLastGive(player, bot, entry, name);
            }
            else
            {
                missing.push_back(name);
            }
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

    // --- Parsing -----------------------------------------------------------

    ParsedCommand ParseRegex(const std::string& msgLower)
    {
        ParsedCommand p;
        std::smatch m;

        // Shift-clicked item link(s): treat as give (or ask if it's a question).
        // Handled first so a linked message opens a trade instead of going to chat.
        std::vector<uint32> links = ExtractItemLinkEntries(msgLower);
        if (!links.empty())
        {
            bool question = msgLower.find('?') != std::string::npos ||
                            msgLower.find("do you have") != std::string::npos ||
                            msgLower.find("have you") != std::string::npos;
            if (question)
            {
                p.intent = BotIntent::AskHave;
                p.itemName = msgLower; // DoAskHave pulls the entry from the link
                return p;
            }
            for (uint32 e : links)
            {
                GiveRequest r;
                r.item = "hitem:" + std::to_string(e); // resolved via the link in DirectGive
                p.gives.push_back(r);
            }
            p.intent = BotIntent::Give;
            return p;
        }

        // general inventory listing: "what do you have in your inventory?", "show me your bags"
        if (msgLower.find("inventory") != std::string::npos ||
            msgLower.find("your bag") != std::string::npos ||
            msgLower.find("in your bag") != std::string::npos ||
            msgLower.find("what do you have") != std::string::npos ||
            msgLower.find("what are you carrying") != std::string::npos)
        {
            p.intent = BotIntent::ListInventory;
            return p;
        }

        // give / hand over / share / trade <stuff>, supporting several items at once:
        // "give me 3 linen cloth and all your mana potions"
        std::regex giveRe(R"(\b(?:give|gimme|hand over|share|trade)\b(.*))");
        if (std::regex_search(msgLower, m, giveRe))
        {
            std::string rest = m[1].str();
            // split into per-item segments on commas and the word "and"
            std::string s = std::regex_replace(rest, std::regex(R"(\band\b)"), ",");
            std::istringstream ss(s);
            std::string seg;
            while (std::getline(ss, seg, ','))
            {
                seg = Trim(seg);
                // A segment is actionable if it names an item, OR is contextual
                // ("more"/"another"/...) which resolves to the last item via memory.
                bool contextual = seg.find("more") != std::string::npos ||
                                  seg.find("another") != std::string::npos ||
                                  seg.find("same") != std::string::npos ||
                                  seg.find("again") != std::string::npos ||
                                  seg.find("extra") != std::string::npos;
                if (ItemQueryTokens(seg).empty() && !contextual) continue;

                GiveRequest r;
                std::smatch nm;
                if (std::regex_search(seg, nm, std::regex(R"(\b(\d+)\b)")))
                    r.quantity = static_cast<uint32>(std::stoul(nm[1].str()));
                else if (seg.find("all") != std::string::npos || seg.find("everything") != std::string::npos)
                    r.all = true;
                else
                    r.quantity = 1;
                r.item = seg;
                p.gives.push_back(r);
            }
            p.intent = p.gives.empty() ? BotIntent::None : BotIntent::Give;
            return p;
        }

        // inventory question: "do you have ...", "got any ...", "have you got ...", or "... ?"
        bool looksQuestion = msgLower.find('?') != std::string::npos ||
                             msgLower.find("do you have") != std::string::npos ||
                             msgLower.find("have you") != std::string::npos ||
                             msgLower.find("got any") != std::string::npos;
        if (looksQuestion)
        {
            std::regex haveRe(R"((?:do you have|have you got|have you|got|have|any)\s+(.+))");
            if (std::regex_search(msgLower, m, haveRe))
            {
                std::string phrase = Trim(m[1].str());
                if (!ItemQueryTokens(phrase).empty())
                {
                    p.intent = BotIntent::AskHave;
                    p.itemName = phrase;
                    return p;
                }
            }
        }

        if (msgLower.find("buff") != std::string::npos) { p.intent = BotIntent::Buff;   return p; }
        if (std::regex_search(msgLower, std::regex(R"(\bfollow\b)"))) { p.intent = BotIntent::Follow; return p; }
        if (std::regex_search(msgLower, std::regex(R"(\bcome\b)")))   { p.intent = BotIntent::Come;   return p; }
        if (msgLower.find("heal") != std::string::npos) { p.intent = BotIntent::Heal; return p; }
        if (std::regex_search(msgLower, std::regex(R"(\b(?:help|assist)\b)"))) { p.intent = BotIntent::Help; return p; }
        return p;
    }

    bool LooksLikeCommand(const std::string& msgLower)
    {
        static const char* cues[] = {
            "heal", "give", "gimme", "buff", "come", "follow", "help", "need",
            "cast", "trade", "share", "hand over", "water", "food", "mana", "drink",
            "have", "got", "any", "potion", "?", "inventory", "bag", "carry", "hitem:"
        };
        for (const char* c : cues)
            if (msgLower.find(c) != std::string::npos)
                return true;
        return false;
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

    void RunLLMFallback(ObjectGuid playerGuid, ObjectGuid botGuid, const std::string& msg)
    {
        std::string prompt =
            "You convert a player's message to a World of Warcraft companion bot into a command. "
            "Reply with ONLY compact JSON, no prose: "
            "{\"intent\":\"heal|give|buff|come|follow|help|askhave|listinventory|none\",\"item\":\"<item name or empty>\",\"quantity\":<integer, 0 means all>}. "
            "Use \"help\" if they call for help or to be rescued/assisted in a fight. "
            "Use \"askhave\" if they ask whether you have a specific item. "
            "Use \"listinventory\" if they ask what you have/carry in general. Use \"none\" if it is not a request. "
            "Message: \"" + msg + "\"";

        std::future<std::string> fut = SubmitQuery(prompt);
        // ponytail: the resolved action runs on this detached thread, same as the
        // module's existing async replies. Regex handles the common commands on the
        // world thread; move this to a main-thread task queue if it proves racy.
        std::thread([playerGuid, botGuid, fut = std::move(fut)]() mutable {
            std::string resp;
            try { resp = fut.get(); } catch (...) { return; }

            size_t a = resp.find('{');
            size_t b = resp.rfind('}');
            if (a == std::string::npos || b == std::string::npos || b <= a) return;

            ParsedCommand cmd;
            try
            {
                nlohmann::json j = nlohmann::json::parse(resp.substr(a, b - a + 1));
                cmd.intent = IntentFromString(j.value("intent", "none"));
                std::string item = j.value("item", "");
                uint32 q = j.value("quantity", 1);
                if (cmd.intent == BotIntent::Give)
                {
                    GiveRequest r;
                    r.item = item;
                    if (q == 0) r.all = true; else r.quantity = q;
                    cmd.gives.push_back(r);
                }
                else
                {
                    cmd.itemName = item;
                }
            }
            catch (...) { return; }

            if (cmd.intent == BotIntent::None) return;

            Player* player = ObjectAccessor::FindPlayer(playerGuid);
            Player* bot = ObjectAccessor::FindPlayer(botGuid);
            if (!player || !bot) return;
            ExecuteParsed(player, bot, cmd);
        }).detach();
    }
}

BotCommandResult TryHandleBotCommand(Player* player, const std::string& msg, Player* whisperTarget)
{
    if (!g_BotCommandEnable) return BotCommandResult::NotHandled;
    if (!player || IsBot(player)) return BotCommandResult::NotHandled;

    std::string trimmed = Trim(msg);
    if (trimmed.empty()) return BotCommandResult::NotHandled;
    std::string lower = ToLower(trimmed);

    bool whisperedToBot = whisperTarget && IsBot(whisperTarget);

    // Require a command cue (heal/give/help/...) for both say and whisper. Without
    // one, this is normal conversation - leave it to the regular chatter path so it
    // is not swallowed or misinterpreted as a command.
    if (!LooksLikeCommand(lower))
        return BotCommandResult::NotHandled;

    Player* bot = whisperedToBot ? whisperTarget : FindNearestCommandableBot(player);
    if (!bot) return BotCommandResult::NotHandled;
    if (bot->GetMap() != player->GetMap()) return BotCommandResult::NotHandled;
    if (bot->GetDistance(player) > g_BotCommandRange) return BotCommandResult::NotHandled;

    ParsedCommand cmd = ParseRegex(lower);
    if (cmd.intent != BotIntent::None)
    {
        ExecuteParsed(player, bot, cmd);
        // Suppress only "give": playerbots opens its own trade window in response
        // to a give request. Other commands display normally.
        return cmd.intent == BotIntent::Give ? BotCommandResult::HandledSuppress
                                             : BotCommandResult::Handled;
    }

    // Regex missed but the message passed the command-cue gate; let the LLM classify.
    if (g_BotCommandLLMFallback)
    {
        RunLLMFallback(player->GetGUID(), bot->GetGUID(), trimmed);
        return BotCommandResult::Handled;
    }

    return BotCommandResult::NotHandled;
}
