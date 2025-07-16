
// src/app/wordwise/wordwise.worker.ts
/// <reference lib="webworker" />

import { WordWiseModel, TransformerModel, FlowNetModel, serializeModel, deserializeModel, AnyModel, VocabData, FitCallbacks, FitOptions } from '@/lib/model';
import { buildTextVocabulary, wordsToInputTensor, wordsToTargetTensor, getWordFromPrediction } from '@/utils/tokenizer';
import { Tensor } from '@/lib/tensor';
import { getCheckpoint, saveCheckpoint, clearCheckpoint } from '@/lib/idb';
import { createSequenceBatches } from '@/utils/batching';

let model: AnyModel | null = null;
let vocabData: VocabData | null = null;
let lossHistory: {epoch: number, loss: number}[] = [];
let stopTrainingFlag = false;


/**
 * Handles messages from the main thread.
 */
self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case 'initialize':
        await initialize(payload);
        break;
      case 'start-training-loop':
        await trainingLoop(payload);
        break;
      case 'stop':
         stopTrainingFlag = true;
        break;
      case 'load-model':
        await loadModel(payload);
        break;
      case 'generate':
        await generate(payload);
        break;
      case 'request-model':
        if (model && vocabData) {
            const modelJson = serializeModel(model, vocabData, lossHistory);
            self.postMessage({ type: 'model-response', payload: { modelJson } });
        }
        break;
      case 'check-for-checkpoint':
        const checkpointJson = await getCheckpoint();
        if (checkpointJson) {
           self.postMessage({ type: 'checkpoint-found' });
        }
        break;
      case 'load-from-checkpoint':
         const cpJson = await getCheckpoint();
         if (cpJson) {
            await loadModel({ modelJson: cpJson });
         }
         break;
      case 'clear-checkpoint':
         await clearCheckpoint();
         break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error), error } });
  }
};


async function loadModel(payload: {modelJson: string}) {
    const loaded = deserializeModel(payload.modelJson);
    model = loaded.model;
    vocabData = loaded.vocabData;
    lossHistory = loaded.lossHistory || [];

    const wordsForSampling = ('vocab' in vocabData && vocabData.vocab) ? vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ', '<eos>'].includes(w) && w.length > 2) : [];
    const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());
    
    self.postMessage({
        type: 'model-loaded',
        payload: {
            architecture: (model as any).getArchitecture(),
            sampleWords: shuffled.slice(0, 4),
            lossHistory: lossHistory
        }
    });
}


/**
 * Initializes the model, vocabulary, and training data.
 */
async function initialize(payload: any) {
  const { modelType: newModelType, textCorpus } = payload;
  vocabData = buildTextVocabulary(textCorpus);
  lossHistory = [];
  
  if (newModelType === 'lstm') {
    const { embeddingDim, hiddenSize } = payload;
    model = new WordWiseModel(vocabData.vocabSize, embeddingDim, hiddenSize);
  } else if (newModelType === 'transformer') {
    const { dModel, numHeads, dff, numLayers, seqLen } = payload;
    model = new TransformerModel(vocabData.vocabSize, seqLen, dModel, numLayers, numHeads, dff);
  } else if (newModelType === 'flownet') {
    const { embeddingDim, numLayers, seqLen } = payload;
    model = new FlowNetModel(vocabData.vocabSize, seqLen, embeddingDim, numLayers);
  } else {
    throw new Error('Unknown initialization type');
  }
  
  const wordsForSampling = vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ', '<eos>'].includes(w) && w.length > 2);
  const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());

  self.postMessage({ 
    type: 'initialized', 
    payload: { 
      type: newModelType,
      vocabSize: vocabData.vocabSize,
      sampleWords: shuffled.slice(0, 4)
    } 
  });
}

/**
 * The main training loop, controlled by the page.
 */
async function trainingLoop(payload: {corpus: string, numEpochs: number, batchSize: number, learningRate: number}) {
    if (!model || !vocabData || !('wordToIndex' in vocabData)) {
      self.postMessage({ type: 'error', payload: { message: 'Model is not initialized.' } });
      return;
    }
    stopTrainingFlag = false;
    
    const { corpus, numEpochs, batchSize, learningRate } = payload;
    const seqLen = (model as any).seqLen || 1;

    // Create sequences once
    const words = corpus.toLowerCase().match(/<eos>|[a-zA-Zа-яА-ЯёЁ]+/g) || [];
    const sequences: number[][] = [];
    if (words.length > seqLen) {
      for (let i = 0; i < words.length - seqLen; i++) {
        sequences.push(words.slice(i, i + seqLen + 1).map(w => vocabData!.wordToIndex.get(w) || 0));
      }
    }
    
    if (sequences.length === 0) {
       self.postMessage({ type: 'error', payload: { message: 'Corpus is too short for the given sequence length.' } });
       return;
    }
    
    const totalBatches = Math.ceil(sequences.length / batchSize);
    let batchCounter = 0;

    for (let epoch = 1; epoch <= numEpochs; epoch++) {
        if (stopTrainingFlag) break;

        // Shuffle sequences each epoch
        for (let i = sequences.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [sequences[i], sequences[j]] = [sequences[j], sequences[i]];
        }
        
        for (let i = 0; i < sequences.length; i += batchSize) {
            if (stopTrainingFlag) break;
            const batchSequences = sequences.slice(i, i + batchSize);
            const actualBatchSize = batchSequences.length;

            const inputData = new Float32Array(actualBatchSize * seqLen);
            const targetData = new Float32Array(actualBatchSize * vocabData.vocabSize);

            for (let j = 0; j < actualBatchSize; j++) {
                const seq = batchSequences[j];
                const inputSeq = seq.slice(0, seqLen);
                const targetWordIndex = seq[seqLen];
                inputData.set(inputSeq, j * seqLen);
                targetData[j * vocabData.vocabSize + targetWordIndex] = 1;
            }

            const batchInputs = new Tensor(inputData, [actualBatchSize, seqLen]);
            const batchTargets = new Tensor(targetData, [actualBatchSize, vocabData.vocabSize]);
            
            const { loss, gradients } = await model.fitSingleBatch(batchInputs, batchTargets, learningRate);

            lossHistory.push({ epoch: batchCounter++, loss });
            
            self.postMessage({
                type: 'progress',
                payload: { epoch: epoch, batch: batchCounter, totalBatches: totalBatches * numEpochs, loss, gradients },
            });
            await new Promise(res => setTimeout(res, 0)); // Yield to message queue
        }
        
        self.postMessage({ type: 'epoch-complete', payload: { epoch } });
        if (!stopTrainingFlag) {
             const checkpointJson = serializeModel(model, vocabData, lossHistory);
             await saveCheckpoint(checkpointJson);
        }
    }

    if (stopTrainingFlag) {
        const modelJson = serializeModel(model, vocabData, lossHistory);
        self.postMessage({ type: 'training-stopped', payload: { epoch: lossHistory.length, modelJson } });
    } else {
        self.postMessage({ type: 'training-complete' });
    }
}


async function generate(payload: {startWord: string, numWords: number, temperature: number}) {
    if (!model || !vocabData || !('wordToIndex' in vocabData)) {
        throw new Error("Text model not ready for generation.");
    }
    const { startWord, numWords, temperature } = payload;
    const { wordToIndex, indexToWord } = vocabData;

    let fullGeneratedText = startWord.toLowerCase().split(' ').filter(Boolean);
    
    // State initialization
    let h: Tensor | undefined;
    let c: Tensor | undefined;
    let flowStates: Tensor[] | undefined;

    for (let i = 0; i < numWords; i++) {
        let chosenWord: string;
        let topPredictions: any[] = [];
        let logits: Tensor;

        if (model instanceof WordWiseModel) {
            const lastWord = fullGeneratedText[fullGeneratedText.length - 1];
            const inputIndex = wordToIndex.get(lastWord) || 0;
            const inputTensor = new Tensor([inputIndex], [1, 1]); // Batch size 1, seq len 1

            const result = model.forward(inputTensor, h, c);
            h = result.h; // Update state for next iteration
            c = result.c; // Update state for next iteration
            logits = result.outputLogits;
            
        } else if (model instanceof FlowNetModel || model instanceof TransformerModel) {
             const seqLen = model.seqLen;
             const currentSequence = fullGeneratedText.slice(-seqLen);
             const currentSequenceIndices = currentSequence.map(w => wordToIndex.get(w) || 0);
             while (currentSequenceIndices.length < seqLen) {
                // Pad with <unk> token index (0)
                currentSequenceIndices.unshift(0); 
            }
            const inputTensor = new Tensor(currentSequenceIndices, [1, seqLen]);

            if (model instanceof FlowNetModel) {
                const result = model.forward(inputTensor, flowStates);
                flowStates = result.newStates;
                logits = result.outputLogits.slice([0, seqLen - 1, 0], [1, 1, vocabData.vocabSize]);
            } else { // Transformer
                const result = model.forward(inputTensor);
                logits = result.outputLogits.slice([0, seqLen - 1, 0], [1, 1, vocabData.vocabSize]);
            }
        } else {
             throw new Error("Unknown model type for generation.");
        }
        
        const reshapedLogits = logits.reshape([1, vocabData.vocabSize]);
        const result = getWordFromPrediction(reshapedLogits, indexToWord, temperature, fullGeneratedText);
        chosenWord = result.chosenWord;
        topPredictions = result.topPredictions;
        
        if (chosenWord === '<eos>') {
            break; // Stop generation if we produce the end-of-sequence token
        }

        if (chosenWord === '<unk>') {
            // Let's not add <unk> to the output, but maybe let it generate something else
            continue;
        }
        
        fullGeneratedText.push(chosenWord);

        self.postMessage({
            type: 'generation-chunk',
            payload: {
                text: fullGeneratedText.join(' '),
                predictions: topPredictions
            }
        });
        await new Promise(resolve => setTimeout(resolve, 50)); 
    }

    self.postMessage({ type: 'generation-complete' });
}


self.postMessage({ type: 'worker-ready' });
