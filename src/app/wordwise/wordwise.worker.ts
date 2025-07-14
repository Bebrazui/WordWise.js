// src/app/wordwise/wordwise.worker.ts
/// <reference lib="webworker" />

import { WordWiseModel, ImageWiseModel, serializeModel, AnyModel, VocabData, FitCallbacks } from '@/lib/model';
import { buildTextVocabulary, wordsToInputTensors, wordsToTargetTensors } from '@/utils/tokenizer';
import { buildImageVocabulary } from '@/utils/image-processor';
import { Tensor } from '@/lib/tensor';

let model: AnyModel | null = null;
let vocabData: VocabData | null = null;
let trainingStopFlag = false; // Note: Stopping mechanism needs to be reimplemented within model.fit
let trainingData: {
    inputs: Tensor[],
    targets: Tensor[],
} | null = null;

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
        // TODO: Implement a more robust stopping mechanism for model.fit
        trainingStopFlag = true;
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error) } });
  }
};

/**
 * Initializes the model, vocabulary, and training data.
 */
async function initialize(payload: any) {
  if (payload.type === 'text') {
    const { textCorpus, embeddingDim, hiddenSize } = payload;
    vocabData = buildTextVocabulary(textCorpus);
    model = new WordWiseModel(vocabData.vocabSize, embeddingDim, hiddenSize);
    
    const words = textCorpus.toLowerCase().match(/[a-zа-яё]+/g) || [];
    trainingData = {
        inputs: wordsToInputTensors(words.slice(0, -1), vocabData.wordToIndex),
        targets: wordsToTargetTensors(words.slice(1), vocabData.wordToIndex, vocabData.vocabSize)
    };
    
    const wordsForSampling = vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2);
    const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());
  
    self.postMessage({ 
      type: 'initialized', 
      payload: { 
        type: 'text',
        vocabSize: vocabData.vocabSize,
        sampleWords: shuffled.slice(0, 4)
      } 
    });
  } else if (payload.type === 'image') {
    const { items, imageSize } = payload;
    vocabData = buildImageVocabulary(items.map((item: any) => item.label));
    model = new ImageWiseModel(vocabData.numClasses, imageSize, imageSize, 3);
    
    const imageTensors = items.map((item: any) => new Tensor(item.pixelData, item.shape));
    const targetTensors = items.map((item: any) => {
        const index = (vocabData as { labelToIndex: Map<string, number> }).labelToIndex.get(item.label)!;
        const oneHot = new Float32Array((vocabData as { numClasses: number }).numClasses).fill(0);
        oneHot[index] = 1;
        return new Tensor(oneHot, [1, (vocabData as { numClasses: number }).numClasses]);
    });

    trainingData = {
        inputs: imageTensors,
        targets: targetTensors
    };

     self.postMessage({ 
      type: 'initialized', 
      payload: { 
        type: 'image',
        numClasses: vocabData.numClasses,
      } 
    });
  } else {
    throw new Error('Unknown initialization type');
  }
}

/**
 * Trains the model using the model.fit() method.
 */
async function train(payload: { numEpochs: number, learningRate: number, batchSize: number, lossHistory: {epoch: number, loss: number}[] }) {
  if (!model || !vocabData || !trainingData) {
    throw new Error('Model is not initialized or no training data is available.');
  }

  const { numEpochs, learningRate, batchSize, lossHistory } = payload;
  
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
    }
  };

  await model.fit(trainingData.inputs, trainingData.targets, {
      epochs: numEpochs,
      batchSize,
      learningRate,
      initialEpoch: lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].epoch + 1 : 0
  }, callbacks);
  
  const modelJson = serializeModel(model, vocabData);
  self.postMessage({ type: 'training-complete', payload: { modelJson } });
}

self.postMessage({ type: 'worker-ready' });
