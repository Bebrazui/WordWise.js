// src/app/wordwise/wordwise.worker.ts
/// <reference lib="webworker" />

import { WordWiseModel, TransformerModel, serializeModel, deserializeModel, AnyModel, VocabData, FitCallbacks } from '@/lib/model';
import { buildTextVocabulary, wordsToInputTensors, wordsToTargetTensors } from '@/utils/tokenizer';
import { Tensor } from '@/lib/tensor';

let model: AnyModel | null = null;
let vocabData: VocabData | null = null;
let stopTrainingFlag = false;
let trainingData: {
    inputs: Tensor[],
    targets: Tensor[],
} | null = null;
let modelType: 'lstm' | 'transformer' = 'transformer';

/**
 * Handles messages from the main thread.
 * Manages initialization, training, and stopping the model.
 */
self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case 'initialize':
        await initialize(payload);
        break;
      case 'train':
        await train(payload);
        break;
      case 'stop':
        stopTrainingFlag = true;
        break;
      case 'load-model':
        await loadModel(payload.modelJson);
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error), error } });
  }
};


async function loadModel(modelJson: string) {
    const loaded = deserializeModel(modelJson);
    model = loaded.model;
    vocabData = loaded.vocabData;
    
    if (model instanceof TransformerModel) {
        modelType = 'transformer';
    } else if (model instanceof WordWiseModel) {
        modelType = 'lstm';
    } else {
        throw new Error("Unknown model type after loading.");
    }
    
    self.postMessage({
        type: 'model-loaded',
        payload: {
            architecture: (model as any).getArchitecture(), // A helper method would be nice here
        }
    });
}


/**
 * Initializes the model, vocabulary, and training data.
 */
async function initialize(payload: any) {
  const { modelType: newModelType, textCorpus } = payload;
  modelType = newModelType;
  vocabData = buildTextVocabulary(textCorpus);
  
  if (modelType === 'lstm') {
    const { embeddingDim, hiddenSize } = payload;
    model = new WordWiseModel(vocabData.vocabSize, embeddingDim, hiddenSize);
     const words = textCorpus.toLowerCase().match(/[a-zа-яё]+/g) || [];
     trainingData = {
        inputs: wordsToInputTensors(words.slice(0, -1), vocabData.wordToIndex),
        targets: wordsToTargetTensors(words.slice(1), vocabData.wordToIndex, vocabData.vocabSize)
     };
  } else if (modelType === 'transformer') {
    const { dModel, numHeads, dff, numLayers, seqLen } = payload;
    model = new TransformerModel(vocabData.vocabSize, seqLen, dModel, numLayers, numHeads, dff);
     const words = textCorpus.toLowerCase().match(/[a-zа-яё]+/g) || [];
      trainingData = {
        inputs: wordsToInputTensors(words, vocabData.wordToIndex),
        targets: wordsToTargetTensors(words, vocabData.wordToIndex, vocabData.vocabSize)
     };
  } else {
    throw new Error('Unknown initialization type');
  }
  
  const wordsForSampling = vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2);
  const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());

  self.postMessage({ 
    type: 'initialized', 
    payload: { 
      type: modelType,
      vocabSize: vocabData.vocabSize,
      sampleWords: shuffled.slice(0, 4)
    } 
  });
}

/**
 * Trains the model using the model.fit() method.
 */
async function train(payload: { numEpochs: number, learningRate: number, batchSize: number, lossHistory: {epoch: number, loss: number}[] }) {
  if (!model || !vocabData || !trainingData) {
    throw new Error('Model is not initialized or no training data is available.');
  }

  const { numEpochs, learningRate, batchSize, lossHistory } = payload;
  stopTrainingFlag = false;
  
  const callbacks: FitCallbacks = {
    onEpochEnd: (log) => {
        self.postMessage({
            type: 'progress',
            payload: {
                epoch: log.epoch,
                loss: log.loss,
                gradients: log.gradients,
                progress: ((log.epoch - (lossHistory.length > 0 ? lossHistory[lossHistory.length-1].epoch : -1)) / numEpochs) * 100,
            },
        });
        return stopTrainingFlag; // Return flag to stop training if true
    }
  };

  await model.fit(trainingData.inputs, trainingData.targets, {
      epochs: numEpochs,
      batchSize,
      learningRate,
      initialEpoch: lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].epoch + 1 : 0
  }, callbacks);
  
  if (stopTrainingFlag) {
       self.postMessage({ type: 'training-stopped', payload: { epoch: lossHistory.length > 0 ? lossHistory[lossHistory.length-1].epoch : 0 } });
       return;
  }
  
  const modelJson = serializeModel(model, vocabData);
  self.postMessage({ type: 'training-complete', payload: { modelJson } });
}

self.postMessage({ type: 'worker-ready' });
