import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const VLLM_REVISION = "c8438a3d40168ce1d9eade0dc15ccbe5d27adb68";
export const VLLM_ASCEND_REVISION = "842b030f8375e630eb639e0560eac7735d04f700";

export const models = [
  { key: "Kimi-K3", url: "https://huggingface.co/moonshotai/Kimi-K3", revision: "f831ab66814297da540d832a5235f8e904f29d06", profile: "kimi", defaultEP: 8, defaultSpecSteps: 7 },
  { key: "Kimi-K3-DSpark", url: "https://huggingface.co/RadixArk/Kimi-K3-DSpark/tree/main", revision: "3c5bac301d9cf392706189d82ed947feca6c2f0f", profile: "kimi_dspark", defaultEP: 1, defaultSpecSteps: 7 },
  { key: "DeepSeek-V4-Flash-0731", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731", revision: "7872f01b1d1fe23eabc4c98b48bffcef5a386062", profile: "deepseek_v4", defaultEP: 8, defaultSpecSteps: 7 },
  { key: "DeepSeek-V4-Pro-0813", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813", revision: "72e1d3230f6c080a530b0a1d46f8eb4602340597", profile: "deepseek_v4", defaultEP: 8, defaultSpecSteps: 7 },
  { key: "GLM-5.3", url: "https://huggingface.co/zai-org/GLM-5.3", revision: "187fb9fff6319062325ff825627ef6db084d9bc6", profile: "glm53", defaultEP: 8, defaultSpecSteps: 1 },
  { key: "Qwen3.8-2.4T-A95B", url: "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B", revision: "207bd685a7e3696cfaff12ded7c6a7ea0f88c996", profile: "qwen38", defaultEP: 8, defaultSpecSteps: 1 },
];

export const modelSlugs = {
  "Kimi-K3": "kimi-k3",
  "Kimi-K3-DSpark": "kimi-k3-dspark",
  "DeepSeek-V4-Flash-0731": "deepseek-v4-flash-0731",
  "DeepSeek-V4-Pro-0813": "deepseek-v4-pro-0813",
  "GLM-5.3": "glm-5.3",
  "Qwen3.8-2.4T-A95B": "qwen3.8-2.4t-a95b",
};

export const defaultConfigRoot = fileURLToPath(new URL("../models/", import.meta.url));
export async function readModelConfig(slug, configRoot = process.env.MODEL_CONFIG_ROOT || defaultConfigRoot) {
  return JSON.parse(await fs.readFile(path.join(configRoot, slug, "config.json"), "utf8"));
}
