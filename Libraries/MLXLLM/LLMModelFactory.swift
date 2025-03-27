import Foundation
import MLX
import MLXLMCommon
import Tokenizers

private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (URL) throws -> M {
    { url in
        let configuration = try JSONDecoder().decode(
            C.self, from: Data(contentsOf: url))
        return modelInit(configuration)
    }
}

public class LLMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {
    public static let shared: LLMTypeRegistry = .init(creators: all())

    private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] {
        [
            "mistral": create(LlamaConfiguration.self, LlamaModel.init),
            "llama": create(LlamaConfiguration.self, LlamaModel.init),
            "phi": create(PhiConfiguration.self, PhiModel.init),
            "phi3": create(Phi3Configuration.self, Phi3Model.init),
            "phimoe": create(PhiMoEConfiguration.self, PhiMoEModel.init),
            "gemma": create(GemmaConfiguration.self, GemmaModel.init),
            "gemma2": create(Gemma2Configuration.self, Gemma2Model.init),
            "qwen2": create(Qwen2Configuration.self, Qwen2Model.init),
            "starcoder2": create(Starcoder2Configuration.self, Starcoder2Model.init),
            "cohere": create(CohereConfiguration.self, CohereModel.init),
            "openelm": create(OpenElmConfiguration.self, OpenELMModel.init),
            "internlm2": create(InternLM2Configuration.self, InternLM2Model.init),
        ]
    }
}

public class LLMRegistry: AbstractModelRegistry, @unchecked Sendable {
    public static let shared = LLMRegistry(modelConfigurations: all())

    static public let smolLM_135M_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/smolLM_135M_4bit"),
        name: "SmolLM-135M-Instruct-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "Tell me about the history of Spain."
    )

    static public let mistralNeMo4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/mistralNeMo4bit"),
        name: "Mistral-Nemo-Instruct-2407-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "Explain quaternions."
    )

    static public let mistral7B4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/mistral7B4bit"),
        name: "Mistral-7B-Instruct-v0.3-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "Describe the Swift language."
    )

    static public let codeLlama13b4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/codeLlama13b4bit"),
        name: "CodeLlama-13b-Instruct-hf-4bit-MLX",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    )

    static public let deepSeekR1_7B_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/deepSeekR1_7B_4bit"),
        name: "DeepSeek-R1-Distill-Qwen-7B-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "Is 9.9 greater or 9.11?"
    )

    static public let phi4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/phi4bit"),
        name: "phi-2-hf-4bit-mlx",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let phi3_5_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/phi3_5_4bit"),
        name: "Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    static public let phi3_5MoE = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/phi3_5MoE"),
        name: "Phi-3.5-MoE-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    static public let gemma2bQuantized = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/gemma2bQuantized"),
        name: "quantized-gemma-2b-it",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_9b_it_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/gemma_2_9b_it_4bit"),
        name: "gemma-2-9b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_2b_it_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/gemma_2_2b_it_4bit"),
        name: "gemma-2-2b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let qwen205b4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/qwen205b4bit"),
        name: "Qwen1.5-0.5B-Chat-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "why is the sky blue?"
    )

    static public let qwen2_5_7b = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/qwen2_5_7b"),
        name: "Qwen2.5-7B-Instruct-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let openelm270m4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/openelm270m4bit"),
        name: "OpenELM-270M-Instruct",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "Once upon a time there was"
    )

    static public let llama3_1_8B_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/llama3_1_8B_4bit"),
        name: "Meta-Llama-3.1-8B-Instruct-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_8B_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/llama3_8B_4bit"),
        name: "Meta-Llama-3-8B-Instruct-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_1B_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com/llama3_2_1B_4bit"),
        name: "Llama-3.2-1B-Instruct-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_3B_4bit = ModelConfiguration(
        source: .url("https://appcaptainshot.frenchpavillon.com"),
        name: "Llama-3.2-3B-Instruct-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    private static func all() -> [ModelConfiguration] {
        [
            smolLM_135M_4bit,
            mistralNeMo4bit,
            mistral7B4bit,
            codeLlama13b4bit,
            deepSeekR1_7B_4bit,
            phi4bit,
            phi3_5_4bit,
            phi3_5MoE,
            gemma2bQuantized,
            gemma_2_9b_it_4bit,
            gemma_2_2b_it_4bit,
            qwen205b4bit,
            qwen2_5_7b,
            openelm270m4bit,
            llama3_1_8B_4bit,
            llama3_8B_4bit,
            llama3_2_1B_4bit,
            llama3_2_3B_4bit
        ]
    }
}

private struct LLMUserInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration

    internal init(tokenizer: any Tokenizer, configuration: ModelConfiguration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    func prepare(input: UserInput) throws -> LMInput {
        do {
            let messages = input.prompt.asMessages()
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(promptTokens))
        } catch {
            let prompt = input.prompt
                .asMessages()
                .compactMap { $0["content"] as? String }
                .joined(separator: ". ")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

public class LLMModelFactory: ModelFactory {
    public init(typeRegistry: ModelTypeRegistry, modelRegistry: AbstractModelRegistry) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared)

    public let typeRegistry: ModelTypeRegistry
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        let modelDirectory = try await downloadModel(
            configuration: configuration, progressHandler: progressHandler)

        let configurationURL = modelDirectory.appending(component: "model.json")
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL))
        let model = try typeRegistry.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType)

        try loadWeights(
            modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization)

        let tokenizer = try await loadTokenizer(configuration: configuration)

        return .init(
            configuration: configuration, model: model,
            processor: LLMUserInputProcessor(tokenizer: tokenizer, configuration: configuration),
            tokenizer: tokenizer)
    }
}
