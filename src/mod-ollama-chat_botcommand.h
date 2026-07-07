#ifndef MOD_OLLAMA_CHAT_BOTCOMMAND_H
#define MOD_OLLAMA_CHAT_BOTCOMMAND_H

#include <cstdint>
#include <string>

class Player;

enum class BotCommandResult
{
    NotHandled,       // not a command; let normal chatter handle it
    Handled,          // handled; skip LLM chatter but let the message display
    HandledSuppress   // handled; also suppress the message (e.g. give, so playerbots
                      // doesn't open its own trade window in response)
};

// Parses a player's natural-language message and, if it is a recognized command
// (heal/give/buff/come/follow/help/inventory) aimed at a nearby bot, performs it.
// whisperTarget is the receiver when the player whispered a specific bot, else nullptr.
BotCommandResult TryHandleBotCommand(Player* player, const std::string& msg, Player* whisperTarget, uint32_t chatType);

// Runs queued bot-command actions on the world thread. Call from a world OnUpdate so
// LLM-resolved actions never mutate game state on a background thread.
void PumpBotCommandTasks();

#endif // MOD_OLLAMA_CHAT_BOTCOMMAND_H
