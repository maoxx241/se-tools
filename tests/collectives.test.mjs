import test from "node:test";
import assert from "node:assert/strict";
import { estimateCollective, estimateAllToAllV, groupEqualAllToAll } from "../lib/collectives.mjs";
import { calculateStepTraffic, loadCommunicationRequirements } from "../lib/communication-requirements.mjs";

test("100 MiB Ring AllReduce over 8 ranks sends 1400 MiB, counted once at sender", () => {
  const r = estimateCollective({ collective: "AllReduce", algorithm: "ring", count: 52_428_800, dtypeBytes: 2, ranks: 8 });
  assert.equal(r.groupSentBytesPerCall, 1400n * 1024n ** 2n);
  assert.equal(r.rankAverageSentBytesPerCall.numerator / 8n, 175n * 1024n ** 2n);
  assert.equal(r.physicalLinkBytes, null);
});

test("AllGather sendCount and ReduceScatter recvCount normalize opposite sides", () => {
  const args = { algorithm: "ring", count: 128, dtypeBytes: 2, ranks: 8, callsPerStep: 3 };
  const ag = estimateCollective({ ...args, collective: "AllGather" });
  const rs = estimateCollective({ ...args, collective: "ReduceScatter" });
  assert.equal(ag.localInputBytes, 256n);
  assert.equal(ag.localOutputBytes, 2048n);
  assert.equal(rs.localInputBytes, 2048n);
  assert.equal(rs.localOutputBytes, 256n);
  assert.equal(rs.groupSentBytesPerStep, 43_008n);
  assert.equal(rs.groupSentBytesPerStep, ag.groupSentBytesPerStep);
});

test("AlltoAll sendCount is per peer; local copy stays out of network payload", () => {
  const r = estimateCollective({ collective: "AlltoAll", algorithm: "direct", count: 10, dtypeBytes: 2, ranks: 4 });
  assert.equal(r.localInputBytes, 80n);
  assert.equal(r.groupSentBytesPerCall, 240n);
  assert.throws(() => groupEqualAllToAll(7, 4), /divisible/);
});

test("AlltoAllV asymmetric matrix retains imbalance and excludes diagonal", () => {
  const input = { sendCounts: [[100, 10, 0], [30, 200, 5], [2, 0, 300]], dtypeBytes: 2 };
  const r = estimateAllToAllV(input);
  assert.deepEqual(r.rankSentBytesPerCall, [20n, 70n, 4n]);
  assert.deepEqual(r.rankReceivedBytesPerCall, [64n, 20n, 10n]);
  assert.deepEqual(r.localCopyBytesPerCall, [200n, 400n, 600n]);
  assert.equal(r.groupSentBytesPerCall, 94n);
  assert.equal(r.maxRankSentBytesPerCall, 70n);
  assert.throws(() => estimateAllToAllV({ ...input, recvCounts: input.sendCounts }), /mismatch/);
});

test("single rank, empty payload, unsafe integers, and unsupported algorithms are explicit", () => {
  for (const collective of ["AllReduce", "AllGather", "ReduceScatter", "AlltoAll"]) {
    const args = { collective, algorithm: collective === "AlltoAll" ? "direct" : "ring", dtypeBytes: 2 };
    assert.equal(estimateCollective({ ...args, count: 15, ranks: 1 }).groupSentBytesPerCall, 0n);
    assert.equal(estimateCollective({ ...args, count: 0, ranks: 8 }).groupSentBytesPerCall, 0n);
  }
  const args = { collective: "AllReduce", algorithm: "ring", ranks: 8, dtypeBytes: 2 };
  assert.throws(() => estimateCollective({ ...args, count: Number.MAX_SAFE_INTEGER + 1 }), /safe integer/);
  assert.equal(estimateCollective({ ...args, count: "9007199254740993" }).localInputBytes, 18014398509481986n);
  assert.throws(() => estimateCollective({ ...args, count: 1, algorithm: "NHR" }), /own model/);
  assert.throws(() => estimateCollective({ ...args, count: 1, ranks: 0 }), /ranks/);
  assert.throws(() => estimateAllToAllV({ sendCounts: [[1], [2]], dtypeBytes: 2 }), /square/);
});

test("non-SP TP attention uses output AllReduce, not SP gather/scatter", () => {
  const facts = { H: 16, attentionLayers: 2, denseLayers: 0, moeLayers: 0 };
  const r = calculateStepTraffic(facts, 8n, 1n, { tp: 4n, ep: 1n, spEnabled: false, dsa: false, sharedExpertTpEnabled: false });
  assert.equal(r.attention, 3072n);
  assert.equal(r.sp, 0n);
});

test("256K case preserves family-specific Decode groups and shared-expert policy", async () => {
  const results = await loadCommunicationRequirements();
  assert.equal(results.length, 6);
  for (const r of results) {
    if (r.dsaModel) {
      assert.deepEqual(r.topology.decode, { tp: 1, dp: 32, ep: 32, sp: false, dsa: false });
      assert.equal(r.decode.attention + r.decode.sp + r.decode.tpBoundary + r.decode.dsaCp, 0n);
      assert.equal(r.decode.epAllToAll * 16n, r.prefill.epAllToAll);
    } else assert.equal(r.topology.decode.tp, 8);
    if (r.slug !== "kimi-k3") assert.equal(r.prefill.sharedExpertTp, 0n);
  }
  assert.equal(results.find((r) => r.slug === "qwen3.8-27b").decode.epAllToAll, 0n);
});
