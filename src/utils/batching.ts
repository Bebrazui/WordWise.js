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
 * Creates batches from sequences of text input and target tensors.
 * This is a critical function for sequence models.
 * @param inputTensors Array of input tensors (word indices).
 * @param targetTensors Array of target tensors (one-hot vectors).
 * @param batchSize The desired batch size.
 * @param seqLen The fixed sequence length for the model.
 * @param useOverlapping For Transformer, sequences can overlap. For RNN, they shouldn't.
 * @returns An array of batches (plain objects for worker compatibility).
 */
export function createSequenceBatches(
    inputTensors: Tensor[],
    targetTensors: Tensor[],
    batchSize: number,
    seqLen: number,
    useOverlapping: boolean = true
): Batch[] {
    const batches: Batch[] = [];
    const totalItems = inputTensors.length;
    
    if (totalItems <= seqLen) {
        console.warn("Not enough data to create even a single sequence.");
        return [];
    }
    
    const vocabSize = targetTensors[0].shape[targetTensors[0].shape.length - 1];
    const step = useOverlapping ? 1 : seqLen;
    
    const sequences: {inputs: number[], targets: number[][]}[] = [];

    // 1. Create all possible sequences from the text
    for (let i = 0; i <= totalItems - seqLen; i += step) {
        const inputSeq = inputTensors.slice(i, i + seqLen).map(t => t.data[0]);
        // The target for a sequence is the same sequence, shifted by one.
        // The model learns to predict the next word at each position.
        const targetSeq = targetTensors.slice(i, i + seqLen).map(t => Array.from(t.data));
        sequences.push({ inputs: inputSeq, targets: targetSeq });
    }

    if (sequences.length === 0) {
        return [];
    }

    // 2. Group sequences into batches
    for (let i = 0; i < sequences.length; i += batchSize) {
        const batchSequences = sequences.slice(i, i + batchSize);
        const actualBatchSize = batchSequences.length;

        const batchInputs = new Float32Array(actualBatchSize * seqLen);
        const batchTargets = new Float32Array(actualBatchSize * seqLen * vocabSize);

        batchSequences.forEach((seq, batchIndex) => {
            batchInputs.set(seq.inputs, batchIndex * seqLen);
            // Flatten the targets for the batch
            const flatTargets = seq.targets.flat();
            batchTargets.set(flatTargets, batchIndex * seqLen * vocabSize);
        });
        
        batches.push({
            inputs: { data: batchInputs, shape: [actualBatchSize, seqLen] },
            targets: { data: batchTargets, shape: [actualBatchSize, seqLen, vocabSize] },
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
  const batches: Batch[] = [];
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
