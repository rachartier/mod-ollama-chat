#ifndef MOD_OLLAMA_CHAT_QUERYMANAGER_H
#define MOD_OLLAMA_CHAT_QUERYMANAGER_H

#include <cstdint>
#include <string>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

// rawMode=true returns the model's text as-is (no chat-style extraction of the text
// between double quotes) - required for the JSON intent call.
// formatJson, when non-empty, is sent as the Ollama "format" field (a JSON schema string)
// to grammar-constrain the output. Requires Ollama >= 0.5.0.
std::string QueryOllamaAPI(const std::string& prompt, uint32_t numPredictOverride = 0, float temperatureOverride = -1.0f, bool rawMode = false, const std::string& formatJson = "");

class QueryManager {
public:
    QueryManager();
    void setMaxConcurrentQueries(int maxQueries);
    std::future<std::string> submitQuery(const std::string& prompt);

private:
    struct QueryTask {
        std::string prompt;
        std::promise<std::string> promise;
    };

    void processQuery(const std::string& prompt, std::promise<std::string> promise);

    int maxConcurrentQueries; // 0 means no limit
    int currentQueries;
    std::mutex mutex_;
    std::queue<QueryTask> taskQueue;
};

#endif // MOD_OLLAMA_CHAT_QUERYMANAGER_H
