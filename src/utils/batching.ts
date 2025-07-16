
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
 * Creates batches for sequence models.
 * @param inputTensors Array of input tensors (sequences of word indices).
 * @param targetTensors Array of target tensors (one-hot vectors for the word after the sequence).
 * @param batchSize The desired batch size.
 * @returns An array of batches (plain objects for worker compatibility).
 */
export function createSequenceBatches(
    inputTensors: Tensor[],
    targetTensors: Tensor[],
    batchSize: number
): Batch[] {
    const batches: Batch[] = [];
    const totalSequences = inputTensors.length;

    if (totalSequences === 0) {
        return [];
    }
    
    for (let i = 0; i < totalSequences; i += batchSize) {
        const batchEnd = Math.min(i + batchSize, totalSequences);
        const actualBatchSize = batchEnd - i;
        
        const inputBatchTensors = inputTensors.slice(i, batchEnd);
        const targetBatchTensors = targetTensors.slice(i, batchEnd);

        // Flatten the data for the batch
        const batchInputsData = new Float32Array(actualBatchSize * inputBatchTensors[0].size);
        const batchTargetsData = new Float32Array(actualBatchSize * targetBatchTensors[0].size);

        for (let j = 0; j < actualBatchSize; j++) {
            batchInputsData.set(inputBatchTensors[j].data, j * inputBatchTensors[j].size);
            batchTargetsData.set(targetBatchTensors[j].data, j * targetBatchTensors[j].size);
        }
        
        batches.push({
            inputs: { data: batchInputsData, shape: [actualBatchSize, ...inputBatchTensors[0].shape] },
            targets: { data: batchTargetsData, shape: [actualBatchSize, ...targetBatchTensors[0].shape] },
        });
    }

    return batches;
}
