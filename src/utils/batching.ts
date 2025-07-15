// src/utils/batching.ts
import { Tensor } from '../lib/tensor';

// Plain object representation of a Tensor for worker communication
type PlainTensor = {
  data: Float32Array;
  shape: number[];
};

type Batch = {
  inputs: PlainTensor;
  targets: PlainTensor;
};

/**
 * Creates batches from sequences of text input and target tensors for Transformer models.
 * @param inputTensors Array of input tensors (word indices).
 * @param targetTensors Array of target tensors (one-hot vectors).
 * @param batchSize The desired batch size.
 * @param seqLen The fixed sequence length for the model.
 * @returns An array of batches (plain objects for worker compatibility).
 */
export function createSequenceBatches(inputTensors: Tensor[], targetTensors: Tensor[], batchSize: number, seqLen: number): Batch[] {
    const batches: Batch[] = [];
    const totalWords = inputTensors.length;
    if (totalWords < seqLen + 1) {
        console.warn("Not enough data to create a single batch.");
        return [];
    }
    
    const vocabSize = targetTensors[0].shape[1];

    // Iterate through the data to create overlapping sequences
    for (let i = 0; i < totalWords - seqLen; i += batchSize * seqLen) {
        
        let currentInputBatch: number[] = [];
        let currentTargetBatch: number[] = [];
        
        for (let b = 0; b < batchSize; b++) {
            const startIdx = i + (b * seqLen);
            if (startIdx + seqLen >= totalWords) break;

            const inputSeq = inputTensors.slice(startIdx, startIdx + seqLen);
            const targetSeq = targetTensors.slice(startIdx + 1, startIdx + seqLen + 1);

            inputSeq.forEach(t => currentInputBatch.push(t.data[0]));
            targetSeq.forEach(t => currentTargetBatch.push(...t.data));
        }
        
        const actualBatchSize = Math.floor(currentInputBatch.length / seqLen);
        if (actualBatchSize === 0) continue;
        
        // Truncate if we overshot
        const finalInputData = new Float32Array(currentInputBatch.slice(0, actualBatchSize * seqLen));
        const finalTargetData = new Float32Array(currentTargetBatch.slice(0, actualBatchSize * seqLen * vocabSize));

        batches.push({
            inputs: { data: finalInputData, shape: [actualBatchSize, seqLen] },
            targets: { data: finalTargetData, shape: [actualBatchSize, seqLen, vocabSize] },
        });
    }

    return batches;
}


/**
 * Creates batches from sequences of image and target tensors.
 * Shuffles the data before batching.
 * @param inputTensors Array of image tensors.
 * @param targetTensors Array of target tensors (one-hot).
 * @param batchSize The desired batch size.
 * @returns An array of batches (plain objects for worker compatibility).
 */
export function createImageBatches(inputTensors: Tensor[], targetTensors: Tensor[], batchSize: number): Batch[] {
  if (inputTensors.length !== targetTensors.length) {
    throw new Error("Input and target tensors must have the same length.");
  }
  const batches = [];
  const totalItems = inputTensors.length;

  // Shuffle data
  const indices = Array.from({ length: totalItems }, (_, i) => i);
  for (let i = totalItems - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  for (let i = 0; i < totalItems; i += batchSize) {
    const batchIndices = indices.slice(i, i + batchSize);
    const actualBatchSize = batchIndices.length;

    if (actualBatchSize === 0) continue;

    const firstInput = inputTensors[batchIndices[0]];
    const firstTarget = targetTensors[batchIndices[0]];

    if (!firstInput || !firstTarget) continue;

    const [channels, height, width] = firstInput.shape;
    const numClasses = firstTarget.shape[1];

    const batchedInputData = new Float32Array(actualBatchSize * channels * height * width);
    const batchedTargetData = new Float32Array(actualBatchSize * numClasses);

    for (let j = 0; j < actualBatchSize; j++) {
      const originalIndex = batchIndices[j];
      const input = inputTensors[originalIndex];
      const target = targetTensors[originalIndex];

      if (input && target) {
        batchedInputData.set(input.data, j * input.size);
        batchedTargetData.set(target.data, j * target.size);
      }
    }

    batches.push({
      inputs: { data: batchedInputData, shape: [actualBatchSize, channels, height, width] },
      targets: { data: batchedTargetData, shape: [actualBatchSize, numClasses] },
    });
  }

  return batches;
}
