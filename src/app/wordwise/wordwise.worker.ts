
// src/app/wordwise/wordwise.worker.ts
/// <reference lib="webworker" />

import { WordWiseModel, TransformerModel, FlowNetModel, serializeModel, deserializeModel, AnyModel, VocabData, FitCallbacks, FitOptions } from '@/lib/model';
import { buildTextVocabulary, wordsToInputTensors, wordsToTargetTensors, getWordFromPrediction } from '@/utils/tokenizer';
import { Tensor } from '@/lib/tensor';
import { getCheckpoint, saveCheckpoint, clearCheckpoint } from '@/lib/idb';

let model: AnyModel | null = null;
let vocabData: VocabData | null = null;
let trainingOptions: (FitOptions & {lossHistory: any[]}) | null = null;
let streamingCorpusBuffer = '';
let lossHistory: {epoch: number, loss: number}[] = [];


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
      case 'train':
        await train(payload);
        break;
      case 'train-stream-start':
        trainStreamStart(payload);
        break;
      case 'train-stream-chunk':
        await trainStreamChunk(payload);
        break;
      case 'train-stream-end':
        await trainStreamEnd();
        break;
      case 'stop':
         if (model) {
            model.stopTraining = true;
         }
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
    const { modelJson } = payload;
    const loaded = deserializeModel(modelJson);
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
 * Trains the model using the model.fit() method with the full corpus.
 */
async function train(payload: { numEpochs: number, learningRate: number, batchSize: number, fullCorpus: string, lossHistory: any[] }) {
  if (!model || !vocabData || !('wordToIndex' in vocabData)) {
    self.postMessage({ type: 'error', payload: { message: 'Model is not initialized or no training data is available.' } });
    return;
  }
  const { numEpochs, learningRate, batchSize, fullCorpus } = payload;
  lossHistory = payload.lossHistory || [];
  const words = (fullCorpus.toLowerCase().match(/<eos>|[a-zA-Zа-яА-ЯёЁ]+/g) || []);
  const trainingData = {
     inputs: wordsToInputTensors(words, vocabData.wordToIndex),
     targets: wordsToTargetTensors(words, vocabData.wordToIndex, vocabData.vocabSize)
  };

  model.stopTraining = false;
  
  const callbacks: FitCallbacks = {
    onEpochEnd: async (log) => {
        lossHistory.push({epoch: log.epoch, loss: log.loss});
        self.postMessage({
            type: 'progress',
            payload: { epoch: log.epoch, loss: log.loss, gradients: log.gradients },
        });
        
        if (model && vocabData) {
            const checkpointJson = serializeModel(model, vocabData, lossHistory);
            await saveCheckpoint(checkpointJson);
        }
        return model?.stopTraining || false;
    }
  };
  
  const fitOptions = {
      epochs: numEpochs,
      batchSize,
      learningRate,
      initialEpoch: lossHistory.length
  };

  await model.fit(trainingData.inputs, trainingData.targets, fitOptions, callbacks);
  
  if (model.stopTraining && model && vocabData) {
       const modelJson = serializeModel(model, vocabData, lossHistory);
       self.postMessage({ type: 'training-stopped', payload: { epoch: lossHistory.length, modelJson } });
       return;
  }
  
  const modelJson = serializeModel(model, vocabData, lossHistory);
  self.postMessage({ type: 'training-complete', payload: { modelJson } });
}


// --- Stream Training Functions ---

function trainStreamStart(payload: FitOptions & {lossHistory: any[]}) {
    if (!model || !vocabData) {
        self.postMessage({ type: 'error', payload: { message: 'Model not initialized for streaming.' }});
        return;
    }
    lossHistory = payload.lossHistory || [];
    trainingOptions = {...payload, initialEpoch: lossHistory.length};
    streamingCorpusBuffer = '';
    model.stopTraining = false;
}

async function trainStreamChunk(payload: { chunk: string }) {
    if (!model || !vocabData || !trainingOptions || !('wordToIndex' in vocabData)) return;

    streamingCorpusBuffer += payload.chunk;

    const words = streamingCorpusBuffer.toLowerCase().match(/<eos>|[a-zA-Zа-яА-ЯёЁ]+/g) || [];
    
    const seqLen = (model as any).seqLen || 1;
    if (words.length < seqLen) return;

    const trainingData = {
       inputs: wordsToInputTensors(words, vocabData.wordToIndex),
       targets: wordsToTargetTensors(words, vocabData.wordToIndex, vocabData.vocabSize)
    };
    
    const leftoverWords = words.length % seqLen;
    streamingCorpusBuffer = words.slice(-leftoverWords).join(' ');

    const callbacks: FitCallbacks = {
        onEpochEnd: async (log) => {
            lossHistory.push({epoch: log.epoch, loss: log.loss});
            self.postMessage({
                type: 'progress',
                payload: { epoch: log.epoch, loss: log.loss, gradients: log.gradients },
            });
             if (model && vocabData) {
                const checkpointJson = serializeModel(model, vocabData, lossHistory);
                await saveCheckpoint(checkpointJson);
            }
            return model?.stopTraining || false;
        }
    };
    
    const streamOptions = { ...trainingOptions, epochs: trainingOptions.epochs + 1 };
    
    await model.fit(trainingData.inputs, trainingData.targets, streamOptions, callbacks);
}

async function trainStreamEnd() {
    if (model?.stopTraining && model && vocabData) {
        const modelJson = serializeModel(model, vocabData, lossHistory);
        self.postMessage({ type: 'training-stopped', payload: { epoch: lossHistory.length, modelJson } });
        return;
    }

    if (model && vocabData) {
      const modelJson = serializeModel(model, vocabData, lossHistory);
      self.postMessage({ type: 'training-complete', payload: { modelJson } });
    }
    trainingOptions = null;
    streamingCorpusBuffer = '';
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
            
        } else if (model instanceof FlowNetModel) {
            const lastWord = fullGeneratedText[fullGeneratedText.length - 1];
            const inputIndex = wordToIndex.get(lastWord) || 0;
            const inputTensor = new Tensor([inputIndex], [1, 1]); // Batch size 1, seq len 1

            const result = model.forward(inputTensor, flowStates);
            flowStates = result.newStates; // Update state for next iteration
            logits = result.outputLogits; // This is [B, S=1, V]
            
        } else if (model instanceof TransformerModel) {
            const currentSequence = fullGeneratedText.slice(-model.seqLen);
            const currentSequenceIndices = currentSequence.map(w => wordToIndex.get(w) || 0);
            while(currentSequenceIndices.length < model.seqLen) {
                currentSequenceIndices.unshift(0); // Pad with <unk> token
            }
            const inputTensor = new Tensor(currentSequenceIndices, [1, model.seqLen]);
            const result = model.forward(inputTensor);
            // Get logits for the very last word in the sequence
            logits = result.outputLogits.slice([0, model.seqLen - 1, 0], [1, 1, vocabData.vocabSize]);
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

        if (chosenWord === '<unk>' || chosenWord === 'вопрос' || chosenWord === 'ответ') {
            if (chosenWord === '<unk>') break; // Stop if we generate unknown token
            continue; // Skip special tokens but continue generating
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
