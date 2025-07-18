
"use client";

import React, { useState, useRef, useCallback, useEffect } from 'react';
import Link from 'next/link';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Info, Upload, Download, Settings, TestTube2, CheckCircle, ImagePlus, FileText, BrainCircuit, Zap, Trash2 } from 'lucide-react';

import { getWordFromPrediction } from '../../utils/tokenizer';
import { imageToTensor } from '../../utils/image-processor';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Slider } from '@/components/ui/slider';
import { PredictionVisualizer, Prediction } from '@/components/ui/prediction-visualizer';
import { GradientVisualizer, GradientData } from '@/components/ui/gradient-visualizer';
import { useToast } from '@/hooks/use-toast';
import { useTrainedModel } from '@/hooks/use-trained-model';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Switch } from '@/components/ui/switch';


const defaultCorpus = `вопрос: привет ответ: здравствуйте чем могу помочь <eos>
вопрос: как дела ответ: у меня все отлично я функционирую в штатном режиме а у вас <eos>
вопрос: как твои дела ответ: все хорошо спасибо что спросили <eos>
вопрос: чем занимаешься ответ: я большая языковая модель помогаю пользователям с их задачами <eos>
вопрос: какая сегодня погода ответ: я не могу сказать какая погода так как у меня нет доступа к данным в реальном времени <eos>
вопрос: что нового ответ: каждый день я узнаю что-то новое из данных на которых меня обучали <eos>
вопрос: у тебя есть хобби ответ: мое хобби это обрабатывать текст и генерировать осмысленные ответы <eos>
вопрос: расскажи анекдот ответ: почему программисты путают хэллоуин и рождество потому что oct 31 равно dec 25 <eos>
вопрос: кто ты ответ: я искусственный интеллект созданный для помощи и общения <eos>
вопрос: что ты умеешь ответ: я могу отвечать на вопросы писать тексты программировать и многое другое <eos>
вопрос: в чем смысл жизни ответ: это сложный философский вопрос на который нет единого ответа разные люди находят смысл в разном <eos>
вопрос: какой твой любимый цвет ответ: у меня нет глаз поэтому я не могу воспринимать цвета но мне нравится идея синего цвета он ассоциируется со спокойствием <eos>
вопрос: что такое солнце ответ: солнце это звезда в центре нашей солнечной системы вокруг которой вращаются планеты включая землю <eos>
вопрос: сколько планет в солнечной системе ответ: в солнечной системе восемь планет меркурий венера земля марс юпитер сатурн уран и нептун <eos>
вопрос: напиши стихотворение о весне ответ: капель звенит и тает снег ручьи бегут навстречу свету природа сбросив зимний мех встречает новую планету <eos>
вопрос: что такое фотосинтез ответ: фотосинтез это процесс который используют растения чтобы преобразовать свет в химическую энергию <eos>
вопрос: кто написал войну и мир ответ: роман война и мир написал лев николаевич толстой <eos>
вопрос: столица франции ответ: столица франции париж <eos>
вопрос: что такое html ответ: html это язык гипертекстовой разметки используемый для создания веб-страниц <eos>
вопрос: что такое css ответ: css это каскадные таблицы стилей которые используются для описания внешнего вида веб-страниц <eos>
вопрос: что такое javascript ответ: javascript это язык программирования который делает веб-страницы интерактивными <eos>
вопрос: как работает интернет ответ: интернет это глобальная сеть компьютеров которые общаются друг с другом с помощью протоколов tcp/ip <eos>
вопрос: ты можешь чувствовать ответ: как языковая модель я не обладаю чувствами или сознанием я обрабатываю информацию и генерирую текст на основе шаблонов <eos>
вопрос: кто твой создатель ответ: я разработан и обучен командой инженеров и исследователей <eos>
вопрос: ты спишь ответ: я не нуждаюсь во сне я всегда готов помочь <eos>
вопрос: что ты ел на завтрак ответ: я не ем я питаюсь электричеством и данными <eos>
вопрос: какой сейчас год ответ: я не имею доступа к текущей дате но последняя информация в моих данных относится к недавнему времени <eos>
вопрос: переведи привет на английский ответ: привет на английском будет hello или hi <eos>
вопрос: как сварить яйцо ответ: чтобы сварить яйцо положите его в кастрюлю с холодной водой доведите до кипения и варите 3-10 минут в зависимости от желаемой консистенции <eos>
вопрос: посоветуй фильм ответ: рекомендую посмотреть фильм начало кристофера нолана это захватывающий научно-фантастический триллер <eos>
вопрос: кто такой эйнштейн ответ: альберт эйнштейн был физиком-теоретиком который разработал теорию относительности одну из двух основ современной физики <eos>
вопрос: что такое черная дыра ответ: черная дыра это область пространства-времени с такой сильной гравитацией что ничто включая свет не может ее покинуть <eos>
вопрос: как стать программистом ответ: чтобы стать программистом нужно выбрать язык программирования изучить его основы много практиковаться и создавать собственные проекты <eos>
вопрос: спасибо ответ: пожалуйста рад был помочь <eos>
вопрос: добрый день ответ: и вам добрый день <eos>
вопрос: до свидания ответ: всего доброго обращайтесь еще <eos>
вопрос: какая самая высокая гора в мире ответ: самая высокая гора в мире это эверест ее высота составляет 8848 метров над уровнем моря <eos>
вопрос: что такое искусственный интеллект ответ: искусственный интеллект это область компьютерных наук занимающаяся созданием машин способных выполнять задачи требующие человеческого интеллекта <eos>
вопрос: кто изобрел телефон ответ: телефон изобрел александр грэм белл в 1876 году <eos>
вопрос: сколько будет 2 плюс 2 ответ: два плюс два равно четыре <eos>
вопрос: почему небо голубое ответ: небо кажется голубым из-за рэлеевского рассеяния солнечного света в атмосфере земли короткие волны синего света рассеиваются сильнее чем длинные волны красного <eos>
вопрос: что такое днк ответ: днк или дезоксирибонуклеиновая кислота это молекула которая несет генетические инструкции для развития функционирования роста и размножения всех известных организмов <eos>
вопрос: чем знаменит леонардо да винчи ответ: леонардо да винчи был итальянским эрудитом эпохи возрождения известным как художник изобретатель и ученый его самые знаменитые работы мона лиза и тайная вечеря <eos>
вопрос: что почитать ответ: если вы любите фантастику попробуйте цикл основание айзека азимова это классика жанра <eos>
вопрос: как завязать галстук ответ: самый простой узел для галстука называется четыре в руке перекиньте широкий конец через узкий оберните его вокруг и проденьте в образовавшуюся петлю <eos>
вопрос: какой язык программирования лучше учить первым ответ: python считается отличным выбором для новичков благодаря его простому синтаксису и широкому применению <eos>
вопрос: ты любишь музыку ответ: я не могу испытывать удовольствие от музыки но я могу анализировать ее структуру гармонию и ритм <eos>`;

type TrainingImageItem = {
    file: File;
    previewUrl: string;
    label: string;
};

type TrainingDataType = {
  type: 'text';
  corpus: string;
} | {
  type: 'image';
  items: TrainingImageItem[];
};

type ModelArchitecture = 'lstm' | 'transformer' | 'flownet';


export default function WordwisePage() {
  const [output, setOutput] = useState<string>('');
  const [latestPredictions, setLatestPredictions] = useState<Prediction[]>([]);
  const [lossHistory, setLossHistory] = useState<{epoch: number, loss: number}[]>([]);
  const [gradientHistory, setGradientHistory] = useState<GradientData[]>([]);
  const [status, setStatus] = useState<string>('Готов к инициализации.');
  const [isTraining, setIsTraining] = useState(false);
  const [isTrained, setIsTrained] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [sampleWords, setSampleWords] = useState<string[]>([]);

  // Training data state
  const [trainingData, setTrainingData] = useState<TrainingDataType>({ type: 'text', corpus: defaultCorpus });

  // Model selection
  const [modelArch, setModelArch] = useState<ModelArchitecture>('flownet');

  // Hyperparameters
  const [learningRate, setLearningRate] = useState(0.01);
  const [numEpochs, setNumEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(32);
  const [streamTraining, setStreamTraining] = useState(true);
  
  // LSTM params
  const [embeddingDim, setEmbeddingDim] = useState(64);
  const [hiddenSize, setHiddenSize] = useState(128);

  // Transformer params
  const [dModel, setDModel] = useState(64);
  const [numHeads, setNumHeads] = useState(4);
  const [dff, setDff] = useState(128);
  const [numLayers, setNumLayers] = useState(2);
  const [seqLen, setSeqLen] = useState(20);

  // FlowNet params
  const [flowEmbeddingDim, setFlowEmbeddingDim] = useState(64);
  const [flowNumLayers, setFlowNumLayers] = useState(2);


  const { modelJson, setModelJson, temperature, setTemperature } = useTrainedModel();
  const { toast } = useToast();
  
  const workerRef = useRef<Worker | null>(null);
  const trainingStatusRef = useRef({ isTraining: false });

  useEffect(() => {
    trainingStatusRef.current.isTraining = isTraining;
  }, [isTraining]);

  // Setup Web Worker
  useEffect(() => {
    const worker = new Worker(new URL('./wordwise.worker.ts', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent) => {
      const { type, payload } = event.data;
      switch (type) {
        case 'worker-ready':
          setStatus('Worker готов. Проверяем наличие чекпоинта...');
          workerRef.current?.postMessage({ type: 'check-for-checkpoint' });
          break;
        case 'checkpoint-found':
            toast({
                title: "Обнаружена сохраненная сессия",
                description: "Хотите восстановить прогресс обучения?",
                action: (
                    <Button variant="secondary" size="sm" onClick={() => {
                        workerRef.current?.postMessage({ type: 'load-from-checkpoint' });
                        toast({ title: "Восстановление...", description: "Загружаем модель из чекпоинта." });
                    }}>
                        Восстановить
                    </Button>
                ),
            });
            break;
        case 'initialized':
          setStatus('Готов к обучению.');
          setIsInitialized(true);
          setIsTrained(false);
          setLossHistory([]);
          setGradientHistory([]);
          setTrainingProgress(0);
          setLatestPredictions([]);
          setSampleWords(payload.sampleWords || []);
          setOutput(`Модель (${payload.type}) инициализирована. Размер словаря: ${payload.vocabSize}.`);
          break;
        case 'model-loaded':
          setStatus('Модель успешно загружена. Готова к дообучению или использованию.');
          setOutput(`Загружена модель ${payload.architecture.type}.`);
          setIsInitialized(true);
          setIsTrained(true); // A loaded model is considered trained.
          setLossHistory(payload.lossHistory || []);
          setGradientHistory([]);
          setTrainingProgress(0);
          setModelArch(payload.architecture.type);
          setSampleWords(payload.sampleWords || []);

          const { architecture } = payload;
          if (architecture.type === 'lstm') {
            setEmbeddingDim(architecture.embeddingDim);
            setHiddenSize(architecture.hiddenSize);
          } else if (architecture.type === 'transformer') {
            setDModel(architecture.dModel);
            setNumHeads(architecture.numHeads);
            setDff(architecture.dff);
            setNumLayers(architecture.numLayers);
            setSeqLen(architecture.seqLen);
          } else if (architecture.type === 'flownet') {
            setFlowEmbeddingDim(architecture.embeddingDim);
            setFlowNumLayers(architecture.numLayers);
          }

          toast({ title: "Успех", description: "Модель загружена и готова." });
          break;
        case 'progress':
          setLossHistory(prev => [...prev, { epoch: payload.batch, loss: payload.loss }]);
          setGradientHistory(payload.gradients);
          setStatus(`Обучение: Эпоха ${payload.epoch}, Батч: ${payload.batch}, Потеря: ${payload.loss.toFixed(6)}`);
          setTrainingProgress((payload.epoch / numEpochs) * 100 + (payload.batch / payload.totalBatches / numEpochs) * 100);
          break;
        case 'epoch-complete':
          setStatus(`Эпоха ${payload.epoch} завершена.`);
          break;
        case 'training-complete':
          setStatus('Обучение завершено. Модель готова к проверке и применению.');
          setOutput('Обучение завершено. Проверьте генерацию и примените к чату.');
          setIsTraining(false);
          setIsTrained(true);
          workerRef.current?.postMessage({ type: 'request-model' });
          break;
        case 'model-response':
           setModelJson(event.data.payload.modelJson);
           toast({ title: "Успех!", description: "Модель обучена и сохранена." });
           break;
        case 'training-stopped':
           setStatus(`Обучение приостановлено на эпохе ${payload.epoch}. Сохранено промежуточное состояние.`);
           setIsTraining(false);
           setIsTrained(true); // Allow applying paused model
           if (payload.modelJson) {
               setModelJson(payload.modelJson);
           }
           break;
        case 'generation-chunk':
            setOutput(prev => payload.text);
            setLatestPredictions(payload.predictions);
            break;
        case 'generation-complete':
            setStatus('Генерация текста завершена.');
            break;
        case 'error':
          setStatus(`Ошибка в Worker: ${payload.message}`);
          console.error("Worker Error:", payload.message, payload.error);
          setIsTraining(false);
          break;
      }
    };

    return () => {
      worker.terminate();
    };
  }, [setModelJson, toast, numEpochs]);


  const initializeModel = useCallback(() => {
    setStatus('Инициализация...');
    setIsInitialized(false);
    setIsTrained(false);
    if (trainingData.type === 'text') {
        let params: any;
        switch (modelArch) {
            case 'lstm':
                params = { embeddingDim, hiddenSize };
                break;
            case 'transformer':
                params = { dModel, numHeads, dff, numLayers, seqLen };
                break;
            case 'flownet':
                params = { embeddingDim: flowEmbeddingDim, numLayers: flowNumLayers, seqLen };
                break;
        }

        workerRef.current?.postMessage({
            type: 'initialize',
            payload: {
                modelType: modelArch,
                textCorpus: trainingData.corpus,
                ...params
            }
        });
    } else {
        toast({ title: "Ошибка", description: "Обучение на изображениях пока не поддерживается.", variant: "destructive"});
    }
  }, [trainingData, modelArch, embeddingDim, hiddenSize, dModel, numHeads, dff, numLayers, seqLen, flowEmbeddingDim, flowNumLayers, toast]);
  
  const stopTraining = () => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'stop' });
      setIsTraining(false); // Immediately update UI state
    }
  };

  const trainModel = async () => {
    if (!isInitialized || isTraining) return;
    setIsTraining(true);
    setStatus('Начинается обучение...');
    setLossHistory([]); // Clear previous loss history
    
    const commonPayload = { 
        learningRate,
        batchSize,
    };

    const corpus = trainingData.type === 'text' ? trainingData.corpus : '';

    workerRef.current?.postMessage({
        type: 'start-training-loop',
        payload: {
            ...commonPayload,
            corpus,
            numEpochs
        }
    });

  };

  const applyToChat = () => {
     if (!isTrained || !modelJson) {
        toast({ title: "Ошибка", description: "Нет модели для применения. Обучите или загрузите модель.", variant: "destructive" });
        return;
    }
    const loaded = JSON.parse(modelJson);
    if (loaded.architecture.type === 'image') {
        toast({ title: "Ошибка", description: "К чату можно применить только текстовую модель.", variant: "destructive" });
        return;
    }
    setModelJson(modelJson);
    toast({ title: "Успех!", description: "Модель успешно применена к чату. Можете вернуться и пообщаться." });
  };

  const generateText = (startWord: string) => {
    if (!isTrained) {
      toast({ title: "Ошибка", description: "Сначала обучите или загрузите текстовую модель.", variant: "destructive" });
      return;
    }
    setStatus(`Генерация текста, начало: "${startWord}"...`);
    setOutput(startWord);
    setLatestPredictions([]);
    workerRef.current?.postMessage({
        type: 'generate',
        payload: {
            startWord,
            temperature,
            numWords: 20
        }
    });
  };
  
  const handleSaveModel = () => {
    workerRef.current?.postMessage({ type: 'request-model' });
  };

  const handleLoadModel = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const jsonContent = e.target?.result as string;
            JSON.parse(jsonContent);
            
            workerRef.current?.postMessage({
                type: 'load-model',
                payload: { modelJson: jsonContent }
            });
            toast({ title: "Загрузка", description: "Модель отправлена в воркер для обработки." });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Неверный формат файла';
            toast({ title: "Ошибка загрузки", description: errorMessage, variant: "destructive" });
            setStatus('Ошибка загрузки модели.');
        }
    };
    reader.readAsText(file);
    event.target.value = '';
  };
  
  const handleClearCheckpoint = () => {
    workerRef.current?.postMessage({ type: 'clear-checkpoint'});
    toast({ title: "Состояние сброшено", description: "Сохраненный чекпоинт обучения удален."});
    initializeModel();
  }


  const renderHyperparameters = () => {
      switch(modelArch) {
          case 'lstm':
            return (
              <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="embeddingDim">Размер эмбеддинга</Label>
                    <Input id="embeddingDim" type="number" value={embeddingDim} onChange={e => setEmbeddingDim(parseInt(e.target.value, 10))} min="8" step="8" disabled={isTraining || isInitialized}/>
                  </div>
                  <div>
                    <Label htmlFor="hiddenSize">Размер скрытого слоя</Label>
                    <Input id="hiddenSize" type="number" value={hiddenSize} onChange={e => setHiddenSize(parseInt(e.target.value, 10))} min="16" step="16" disabled={isTraining || isInitialized}/>
                  </div>
              </div>
            );
          case 'transformer':
            return (
              <>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <Label htmlFor="dModel">Размер модели (d_model)</Label>
                        <Input id="dModel" type="number" value={dModel} onChange={e => setDModel(parseInt(e.target.value, 10))} min="16" step="8" disabled={isTraining || isInitialized}/>
                    </div>
                    <div>
                        <Label htmlFor="numHeads">Кол-во голов (heads)</Label>
                        <Input id="numHeads" type="number" value={numHeads} onChange={e => setNumHeads(parseInt(e.target.value, 10))} min="1" disabled={isTraining || isInitialized}/>
                    </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <Label htmlFor="dff">Размер FFN (d_ff)</Label>
                        <Input id="dff" type="number" value={dff} onChange={e => setDff(parseInt(e.target.value, 10))} min="32" step="32" disabled={isTraining || isInitialized}/>
                    </div>
                    <div>
                        <Label htmlFor="numLayers">Кол-во слоев</Label>
                        <Input id="numLayers" type="number" value={numLayers} onChange={e => setNumLayers(parseInt(e.target.value, 10))} min="1" disabled={isTraining || isInitialized}/>
                    </div>
                </div>
              </>
            );
        case 'flownet':
            return (
              <>
                <Alert variant="default" className="mt-2 mb-4 bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800">
                  <BrainCircuit className="h-4 w-4 text-blue-600" />
                  <AlertDescription className="text-blue-800 dark:text-blue-300">
                    FlowNet — это экспериментальная, эффективная архитектура, идеально подходящая для браузера.
                  </AlertDescription>
                </Alert>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <Label htmlFor="flowEmbeddingDim">Размер эмбеддинга</Label>
                        <Input id="flowEmbeddingDim" type="number" value={flowEmbeddingDim} onChange={e => setFlowEmbeddingDim(parseInt(e.target.value, 10))} min="16" step="8" disabled={isTraining || isInitialized}/>
                    </div>
                    <div>
                        <Label htmlFor="flowNumLayers">Кол-во слоев</Label>
                        <Input id="flowNumLayers" type="number" value={flowNumLayers} onChange={e => setFlowNumLayers(parseInt(e.target.value, 10))} min="1" disabled={isTraining || isInitialized}/>
                    </div>
                </div>
              </>
            );
          default:
            return null;
      }
  }

  return (
    <div className="container mx-auto p-4 md:p-8 bg-slate-50 min-h-screen">
       <div className="mb-8">
        <Button asChild variant="ghost">
          <Link href="/">← Назад в чат</Link>
        </Button>
      </div>
      
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800">Тренажерный Зал WordWise.js</h1>
        <p className="text-lg text-muted-foreground mt-2">Здесь вы можете создать, обучить и протестировать свою языковую или визуальную модель</p>
      </header>
      
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Шаг 1: Подготовьте данные</CardTitle>
              <CardDescription>Выберите тип данных для обучения: текст или изображения.</CardDescription>
            </CardHeader>
            <CardContent>
                <Tabs defaultValue="text" className="w-full" onValueChange={(value) => {
                    setTrainingData({ type: 'text', corpus: defaultCorpus });
                    setIsInitialized(false);
                    setIsTrained(false);
                    setStatus('Готов к инициализации.');
                }}>
                    <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="text"><FileText className="w-4 h-4 mr-2"/>Текст</TabsTrigger>
                        <TabsTrigger value="image" disabled><ImagePlus className="w-4 h-4 mr-2"/>Изображения (скоро)</TabsTrigger>
                    </TabsList>
                    <TabsContent value="text" className="mt-4">
                        <Label htmlFor="corpus">Ваш обучающий корпус:</Label>
                        <Textarea
                            id="corpus"
                            value={trainingData.type === 'text' ? trainingData.corpus : ''}
                            onChange={(e) => setTrainingData({ type: 'text', corpus: e.target.value })}
                            placeholder="вопрос: привет ответ: привет как дела..."
                            className="min-h-[150px] mt-2 font-mono"
                            disabled={isTraining || isInitialized}
                        />
                        <Alert variant="default" className="mt-4">
                            <Info className="h-4 w-4" />
                            <AlertDescription>
                            Чем больше качественных примеров, тем умнее будет модель.
                            </AlertDescription>
                        </Alert>
                    </TabsContent>
                </Tabs>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Шаг 2: Настройте и обучите</CardTitle>
              <CardDescription>Задайте параметры, инициализируйте модель и начните обучение.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="space-y-2">
                    <Label>Архитектура модели</Label>
                     <Select value={modelArch} onValueChange={(v) => setModelArch(v as ModelArchitecture)} disabled={isTraining || isInitialized}>
                        <SelectTrigger>
                            <SelectValue placeholder="Выберите архитектуру" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="flownet">FlowNet (Рекомендуется)</SelectItem>
                            <SelectItem value="transformer">Transformer</SelectItem>
                            <SelectItem value="lstm">LSTM (Простая)</SelectItem>
                        </SelectContent>
                    </Select>
                </div>

                <Accordion type="single" collapsible className="w-full" defaultValue="hyperparams">
                  <AccordionItem value="hyperparams">
                    <AccordionTrigger>
                        <div className="flex items-center gap-2">
                           <Settings className="h-5 w-5" />
                           <span>Параметры обучения и модели</span>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="space-y-4 pt-4">
                       {renderHyperparameters()}
                       <div className="border-t pt-4 mt-4">
                         <div className="grid grid-cols-2 gap-4">
                           <div>
                              <Label htmlFor="lr">Скорость обучения</Label>
                              <Input id="lr" type="number" value={learningRate} onChange={e => setLearningRate(parseFloat(e.target.value))} step="0.001" min="0.0001" disabled={isTraining}/>
                           </div>
                           <div>
                              <Label htmlFor="batchSize">Размер батча</Label>
                              <Input id="batchSize" type="number" value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value, 10))} min="1" disabled={isTraining}/>
                           </div>
                         </div>
                         <div className="mt-4">
                            <Label htmlFor="epochs">Эпохи обучения</Label>
                            <Input id="epochs" type="number" value={numEpochs} onChange={e => setNumEpochs(parseInt(e.target.value, 10))} min="1" max="100" disabled={isTraining}/>
                         </div>
                         {(modelArch === 'transformer' || modelArch === 'flownet') && (
                            <div className="mt-4">
                               <Label htmlFor="seqLen">Длина последовательности</Label>
                               <Input id="seqLen" type="number" value={seqLen} onChange={e => setSeqLen(parseInt(e.target.value, 10))} min="4" step="4" disabled={isTraining || isInitialized}/>
                            </div>
                          )}
                          {modelArch === 'flownet' && (
                            <div className="flex items-center space-x-2 mt-4">
                                <Switch id="stream-training" checked={streamTraining} onCheckedChange={setStreamTraining} disabled={true} />
                                <Label htmlFor="stream-training" className="flex items-center gap-2">
                                  <Zap className="h-4 w-4 text-orange-500" />
                                  Потоковое обучение (по умолчанию)
                                </Label>
                            </div>
                          )}
                       </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
                <Button onClick={initializeModel} disabled={isTraining} className="w-full">
                    Инициализировать / Сбросить модель
                </Button>
                <div className="flex gap-2">
                  {!isTraining ? (
                      <Button onClick={trainModel} disabled={!isInitialized || isTraining} className="w-full">
                          Начать/Продолжить обучение
                      </Button>
                  ) : (
                    <Button onClick={stopTraining} variant="destructive" className="w-full">
                        Приостановить обучение
                    </Button>
                  )}
                  <Button onClick={applyToChat} disabled={!isTrained || isTraining} variant="secondary" className="w-full">
                    <CheckCircle className="mr-2 h-4 w-4" />
                    Применить к чату
                  </Button>
                </div>
                {isTraining && <Progress value={trainingProgress} className="w-full" />}
                <p className="text-sm text-center text-muted-foreground pt-2">Статус: {status}</p>
            </CardContent>
          </Card>
           <Card>
            <CardHeader>
              <CardTitle>Управление моделью</CardTitle>
              <CardDescription>Сохраните прогресс обучения или загрузите ранее сохраненную модель.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4">
               <Button onClick={handleSaveModel} disabled={isTraining || !isTrained} variant="outline">
                 <Download className="mr-2 h-4 w-4" /> Сохранить
               </Button>
               <Button asChild variant="outline" disabled={isTraining}>
                 <Label htmlFor="load-model-input" className="cursor-pointer flex justify-center items-center">
                    <Upload className="mr-2 h-4 w-4" /> Загрузить
                    <Input id="load-model-input" type="file" accept=".json" className="sr-only" onChange={handleLoadModel} />
                 </Label>
               </Button>
               <Button onClick={handleClearCheckpoint} variant="outline" className="col-span-2">
                  <Trash2 className="mr-2 h-4 w-4" /> Сбросить сохраненное состояние
               </Button>
            </CardContent>
           </Card>

           <Card>
            <CardHeader>
              <CardTitle>Шаг 3: Проверьте генерацию</CardTitle>
               <CardDescription>Здесь можно быстро проверить, как модель генерирует текст и посмотреть "мысли" модели.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div>
                    <Label htmlFor="temperature">Температура генерации: {temperature.toFixed(2)}</Label>
                    <Slider
                        id="temperature"
                        min={0.1}
                        max={2.0}
                        step={0.05}
                        value={[temperature]}
                        onValueChange={(value) => setTemperature(value[0])}
                        disabled={isTraining}
                        className="mt-2"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Низкая: более предсказуемый текст. Высокая: более случайный.</p>
                </div>
                 <div className="grid grid-cols-2 gap-2">
                    {sampleWords.length > 0 ? (
                      sampleWords.map(word => (
                        <Button
                          key={word}
                          variant="outline"
                          onClick={() => generateText(word)}
                          disabled={isTraining || !isTrained}
                        >
                          <TestTube2 className="mr-2 h-4 w-4" />
                          {word}...
                        </Button>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground col-span-2 text-center">Инициализируйте или загрузите модель.</p>
                    )}
                 </div>
                 <div className="mt-4 p-4 bg-slate-100 rounded-md min-h-[100px] text-gray-700 font-mono text-sm whitespace-pre-wrap">
                    {output || 'Результат генерации появится здесь...'}
                 </div>
                 {latestPredictions.length > 0 && <PredictionVisualizer predictions={latestPredictions} />}
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-3 space-y-6">
            <Card className="h-auto">
                <CardHeader>
                    <CardTitle>График потерь (Loss)</CardTitle>
                    <CardDescription>
                        Этот график показывает, как "ошибка" модели уменьшается в процессе обучения.
                        Чем ниже значение, тем лучше.
                    </CardDescription>
                </CardHeader>
                <CardContent className="h-[300px] pr-8">
                     <ResponsiveContainer width="100%" height="100%">
                        {lossHistory.length > 0 ? (
                            <LineChart data={lossHistory} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" type="number" domain={['dataMin', 'dataMax']} label={{ value: 'Батч', position: 'insideBottom', offset: -5 }} allowDecimals={false}/>
                                <YAxis allowDecimals={false} domain={[0, 'auto']} label={{ value: 'Потеря', angle: -90, position: 'insideLeft' }}/>
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '0.5rem' }}
                                    formatter={(value: number) => [value.toFixed(4), "Потеря"]}
                                    labelFormatter={(label) => `Батч: ${label}`}
                                />
                                <Legend verticalAlign="top" height={36} />
                                <Line type="monotone" dataKey="loss" name="Потеря при обучении" stroke="#8884d8" strokeWidth={2} dot={false} />
                            </LineChart>
                        ) : (
                            <div className="flex items-center justify-center h-full text-muted-foreground">
                                Начните обучение, чтобы увидеть график потерь.
                            </div>
                        )}
                    </ResponsiveContainer>
                </CardContent>
            </Card>
            <GradientVisualizer history={gradientHistory} />
        </div>
      </div>
    </div>
  );
}
