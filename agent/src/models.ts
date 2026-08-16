import {
  createModels,
  createProvider,
  type Model,
  type Models,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openrouterProvider } from "@earendil-works/pi-ai/providers/openrouter";
import { resolve } from "node:path";

export type ModelTarget = "slm" | "deepseek";

const DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-pro-0813";

export async function loadModel(target: ModelTarget): Promise<{
  models: Models;
  model: Model<any>;
}> {
  const models = createModels();
  if (target === "deepseek") {
    if (!process.env.OPENROUTER_API_KEY) {
      throw new Error("OPENROUTER_API_KEY is not set in .env");
    }
    models.setProvider(openrouterProvider());
    await models.refresh({ providers: ["openrouter"], force: true });
    const modelId = process.env.OPENROUTER_MODEL?.trim() || DEFAULT_OPENROUTER_MODEL;
    const model = models.getModel("openrouter", modelId);
    if (!model) throw new Error(`OpenRouter did not publish ${modelId}`);
    return { models, model };
  }

  const localModel: Model<"openai-completions"> = {
    id: resolve("models/LFM2.5-2.6B-MLX-6bit"),
    name: "LFM2.5 Spanish Wordle",
    api: "openai-completions",
    provider: "mlx-local",
    baseUrl: "http://127.0.0.1:8080/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 131072,
    maxTokens: 512,
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    },
  };
  models.setProvider(
    createProvider({
      id: "mlx-local",
      name: "MLX Local",
      baseUrl: localModel.baseUrl,
      auth: {
        apiKey: {
          name: "MLX Local",
          resolve: async () => ({
            auth: { apiKey: "mlx-local" },
            source: "local",
          }),
        },
      },
      models: [localModel],
      api: openAICompletionsApi(),
    }),
  );
  const model = models.getModel("mlx-local", localModel.id);
  if (!model) throw new Error("failed to register the local MLX model");
  return { models, model };
}
