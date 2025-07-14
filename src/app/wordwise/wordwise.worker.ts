
// src/app/wordwise/wordwise.worker.ts
/// <reference lib="webworker" />

import { WordWiseModel, ImageWiseModel, serializeModel, BaseModel, VocabData } from '@/lib/model';
import { SGD } from '@/lib/optimizer';
import { crossEntropyLossWithSoftmaxGrad } from '@/lib/layers';
import { buildTextVocabulary, wordsToInputTensors, wordsToTargetTensors, createTextBatches } from '@/utils/tokenizer';
import { buildImageVocabulary, createImageBatches } from '@/utils/image-processor';
import { Tensor } from '@/lib/tensor';

let model: BaseModel | null = null;
let optimizer: SGD | null = null;
let vocabData: VocabData | null = null;
let trainingStopFlag = false;
let trainingData: {
    inputs: Tensor[],
    targets: Tensor[],
} | null = null;

/**
 * Обработчик сообщений от основного потока.
 * Управляет инициализацией, обучением и остановкой модели.
 */
self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case 'initialize':
        await initialize(payload);
        break;
      case 'train':
        trainingStopFlag = false;
        await train(payload);
        break;
      case 'stop':
        trainingStopFlag = true;
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error) } });
  }
};

/**
 * Инициализирует модель, словарь и оптимизатор.
 */
async function initialize(payload: any) {
  const { learningRate } = payload;
  optimizer = new SGD(learningRate);
  
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
    const { items, imageSize } = payload; // `items` now contains { pixelData, shape, label }
    vocabData = buildImageVocabulary(items.map((item: any) => item.label));
    model = new ImageWiseModel(vocabData.numClasses, imageSize, imageSize, 3); // Assuming 3 channels (RGB)
    
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
 * Асинхронно обучает модель.
 */
async function train(payload: { numEpochs: number, learningRate: number, batchSize: number, lossHistory: {epoch: number, loss: number}[] }) {
  if (!model || !vocabData || !optimizer || !trainingData) {
    throw new Error('Модель не инициализирована или нет данных для обучения.');
  }

  const { numEpochs, learningRate, batchSize, lossHistory } = payload;
  optimizer.learningRate = learningRate;

  const batches = model.type === 'text'
    ? createTextBatches(trainingData.inputs, trainingData.targets, batchSize)
    : createImageBatches(trainingData.inputs, trainingData.targets, batchSize);

  if (batches.length === 0) {
    throw new Error("Не удалось создать батчи для обучения. Проверьте данные.");
  }

  const startEpoch = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].epoch + 1 : 0;
  
  for (let epoch = 0; epoch < numEpochs; epoch++) {
    if (trainingStopFlag) {
      self.postMessage({ type: 'training-stopped', payload: { epoch: startEpoch + epoch } });
      return;
    }

    let epochLoss = 0;
    
    for (const batch of batches) {
       // Re-create Tensors from plain objects
      const batchInputs = new Tensor(batch.inputs.data, batch.inputs.shape);
      const batchTargets = new Tensor(batch.targets.data, batch.targets.shape);

      let predictionLogits;
      if (model.type === 'text') {
        let {h0: h, c0: c} = (model as WordWiseModel).initializeStates(batchInputs.shape[0]);
        predictionLogits = model.forward(batchInputs, h, c).outputLogits;
      } else { // image model
        predictionLogits = (model as ImageWiseModel).forward(batchInputs).outputLogits;
      }
      
      const lossTensor = crossEntropyLossWithSoftmaxGrad(predictionLogits, batchTargets);
      // Ensure loss is a single number before adding
      if(lossTensor.data.length === 1) {
        epochLoss += lossTensor.data[0];
      }
      
      lossTensor.backward();
      optimizer.step(model.getParameters());
    }

    const avgEpochLoss = epochLoss / batches.length;
    const currentEpochNumber = startEpoch + epoch;

    self.postMessage({
      type: 'progress',
      payload: {
        epoch: currentEpochNumber,
        loss: avgEpochLoss,
        progress: ((epoch + 1) / numEpochs) * 100,
      },
    });

    await new Promise(resolve => setTimeout(resolve, 10));
  }
  
  const modelJson = serializeModel(model, vocabData);
  self.postMessage({ type: 'training-complete', payload: { modelJson } });
}

self.postMessage({ type: 'worker-ready' });
