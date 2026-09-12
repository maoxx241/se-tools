// HCCL API count semantics and explicitly selected communication models.
// Sources and derivation: analysis/HCCL.md. No hardware/runtime dependencies.
export const HCCL_REVISION = "170ddeec539b4d693028ce6e0cf5c58933e4d46d";

export function integer(value, name, minimum = 0n) {
  if (typeof value === "number" && !Number.isSafeInteger(value)) {
    throw new Error(`${name} must be a safe integer; use a decimal string for large counts`);
  }
  if (!["bigint", "number", "string"].includes(typeof value)
      || (typeof value === "string" && !/^\d+$/.test(value))) {
    throw new Error(`${name} must be an integer`);
  }
  const result = BigInt(value);
  if (result < minimum) throw new Error(`${name} must be >= ${minimum}`);
  return result;
}

function groupInputs(bytes, ranks) {
  return [integer(bytes, "local input bytes"), integer(ranks, "ranks", 1n)];
}

// B is the full local input buffer, except AllGather where it is the local shard.
// Ring results are ideal algorithm payload sums, not measured link counters.
export function groupAllReduce(bytes, ranks) {
  const [b, p] = groupInputs(bytes, ranks);
  return b * 2n * (p - 1n);
}
export function groupAllGather(bytes, ranks) {
  const [b, p] = groupInputs(bytes, ranks);
  return b * p * (p - 1n);
}
export function groupReduceScatter(bytes, ranks) {
  const [b, p] = groupInputs(bytes, ranks);
  return b * (p - 1n);
}
export function groupEqualAllToAll(bytes, ranks) {
  const [b, p] = groupInputs(bytes, ranks);
  if (b % p) throw new Error("Equal-split AlltoAll input bytes must be divisible by ranks");
  return b * (p - 1n);
}

// The workbook stores local INPUT payloads, not HCCL API count arguments.
export function sendRatioFormula(row, collective) {
  const p = `G${row}`;
  const ratios = {
    "Ring AllReduce": `2*(${p}-1)/${p}`,
    "Ring AllGather": `${p}-1`,
    "Ring ReduceScatter": `(${p}-1)/${p}`,
    "Equal-split AllToAll": `(${p}-1)/${p}`,
    "AllToAllV（跨 Rank 发送）": "1",
  };
  if (!(collective in ratios)) throw new Error(`Unknown collective: ${collective}`);
  return `IF(${p}>1,${ratios[collective]},0)`;
}

/** count is the HCCL API count: AR count, AG sendCount, RS recvCount, A2A per-peer sendCount. */
export function estimateCollective({ collective, count, dtypeBytes, ranks, callsPerStep = 1, algorithm }) {
  const p = integer(ranks, "ranks", 1n);
  const c = integer(count, "count");
  const width = integer(dtypeBytes, "dtypeBytes", 1n);
  const calls = integer(callsPerStep, "callsPerStep");
  const b = c * width;
  let input, output, sent, basis, countMeaning;
  if (collective === "AlltoAll") {
    if (algorithm !== "direct") throw new Error("AlltoAll requires algorithm=direct (endpoint payload)");
    input = output = b * p;
    sent = groupEqualAllToAll(input, p);
    basis = "cross-rank endpoint payload; self-copy excluded";
    countMeaning = "sendCount per destination rank";
  } else {
    if (algorithm !== "ring") throw new Error("Select algorithm=ring explicitly; other algorithms require their own model");
    basis = "ideal Ring algorithm payload";
    if (collective === "AllReduce") {
      input = output = b; sent = groupAllReduce(b, p); countMeaning = "full local input/output count";
    } else if (collective === "AllGather") {
      input = b; output = b * p; sent = groupAllGather(b, p); countMeaning = "local input sendCount";
    } else if (collective === "ReduceScatter") {
      input = b * p; output = b; sent = groupReduceScatter(input, p); countMeaning = "local output recvCount";
    } else throw new Error(`Unsupported collective: ${collective}`);
  }
  return {
    collective, algorithm, basis, countMeaning, ranks: p, count: c, dtypeBytes: width, callsPerStep: calls,
    localInputBytes: input, localOutputBytes: output,
    groupSentBytesPerCall: sent, groupSentBytesPerStep: sent * calls,
    // This is an average, not a claim of identical per-rank load with uneven slices.
    rankAverageSentBytesPerCall: { numerator: sent, denominator: p },
    physicalLinkBytes: null,
  };
}

/** Matrix entry [src][dst] is HCCL sendCounts[dst] on src, in elements. */
export function estimateAllToAllV({ sendCounts, dtypeBytes, recvCounts, callsPerStep = 1 }) {
  if (!Array.isArray(sendCounts) || !sendCounts.length) throw new Error("sendCounts must be a nonempty square matrix");
  const p = sendCounts.length;
  const matrix = sendCounts.map((row, r) => {
    if (!Array.isArray(row) || row.length !== p) throw new Error("sendCounts must be a square matrix");
    return row.map((v, c) => integer(v, `sendCounts[${r}][${c}]`));
  });
  const width = integer(dtypeBytes, "dtypeBytes", 1n);
  const calls = integer(callsPerStep, "callsPerStep");
  if (recvCounts !== undefined) {
    if (!Array.isArray(recvCounts) || recvCounts.length !== p) throw new Error("recvCounts must be a square matrix");
    for (let dst = 0; dst < p; dst++) {
      if (!Array.isArray(recvCounts[dst]) || recvCounts[dst].length !== p) throw new Error("recvCounts must be a square matrix");
      for (let src = 0; src < p; src++) {
        if (integer(recvCounts[dst][src], `recvCounts[${dst}][${src}]`) !== matrix[src][dst]) {
          throw new Error(`send/receive count mismatch: ${src} -> ${dst}`);
        }
      }
    }
  }
  const rankSentBytes = matrix.map((row, src) => row.reduce((s, v, dst) => s + (src === dst ? 0n : v * width), 0n));
  const rankReceivedBytes = matrix.map((_, dst) => matrix.reduce((s, row, src) => s + (src === dst ? 0n : row[dst] * width), 0n));
  const groupSentBytes = rankSentBytes.reduce((a, b) => a + b, 0n);
  return {
    collective: "AlltoAllV", basis: "cross-rank endpoint payload; measured counts required for measured routing",
    ranks: p, dtypeBytes: width, callsPerStep: calls,
    rankSentBytesPerCall: rankSentBytes, rankReceivedBytesPerCall: rankReceivedBytes,
    localCopyBytesPerCall: matrix.map((row, r) => row[r] * width),
    groupSentBytesPerCall: groupSentBytes, groupSentBytesPerStep: groupSentBytes * calls,
    maxRankSentBytesPerCall: rankSentBytes.reduce((a, b) => a > b ? a : b, 0n),
    physicalLinkBytes: null,
  };
}

export const jsonBytes = (value) => JSON.stringify(value, (_, v) => typeof v === "bigint" ? v.toString() : v, 2);
