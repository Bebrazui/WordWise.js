
"use client";
import { useState, useRef, useEffect, FormEvent } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { BrainCircuit, Bot, Eye, EyeOff } from 'lucide-react';
import Link from 'next/link';
import { PredictionVisualizer, Prediction } from '@/components/ui/prediction-visualizer';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

import { useTrainedModel } from '@/hooks/use-trained-model';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [latestPredictions, setLatestPredictions] = useState<Prediction[]>([]);
  const [showPredictions, setShowPredictions] = useState(true);
  const { modelJson, temperature } = useTrainedModel();
  const workerRef = useRef<Worker | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const modelName = modelJson ? `WordWise.js (${JSON.parse(modelJson).architecture.type})` : "WordWise.js (Прототип)";

  // Setup Web Worker for chat generation
  useEffect(() => {
    const worker = new Worker(new URL('./wordwise/wordwise.worker.ts', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent) => {
      const { type, payload } = event.data;
      switch (type) {
        case 'worker-ready':
           if (modelJson) {
              console.log("Chat worker ready. Loading model from store...");
              worker.postMessage({ type: 'load-model', payload: { modelJson } });
           }
           break;
        case 'generation-chunk':
           setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];
                if (lastMessage && lastMessage.role === 'assistant') {
                    lastMessage.content = payload.text;
                    if (payload.predictions) {
                        setLatestPredictions(payload.predictions);
                    }
                }
                return newMessages;
            });
            break;
        case 'generation-complete':
            setIsLoading(false);
            if (!showPredictions) {
                setLatestPredictions([]);
            }
            break;
        case 'model-loaded':
            console.log("Model loaded into chat worker.");
            break;
        case 'error':
            setIsLoading(false);
            setMessages(prev => [...prev, { role: 'assistant', content: `Произошла ошибка в воркере: ${payload.message}` }]);
            break;
      }
    };
    
    return () => {
      worker.terminate();
    };
  }, [modelJson, showPredictions]); // Re-setup worker if model changes

  useEffect(() => {
    if (scrollAreaRef.current) {
        const viewport = scrollAreaRef.current.querySelector('div');
        if (viewport) {
            viewport.scrollTop = viewport.scrollHeight;
        }
    }
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    if (modelJson) {
      // Create an empty message shell for the assistant
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
      workerRef.current?.postMessage({
        type: 'generate',
        payload: {
          startWord: currentInput,
          temperature,
          numWords: 20
        }
      });
    } else {
        const botResponseContent = `Я прототип. Модель WordWise.js еще не обучена. Перейдите в раздел обучения, чтобы научить меня отвечать.`;
        const botResponse: ChatMessage = {
            role: 'assistant',
            content: botResponseContent
        };
        setMessages(prev => [...prev, botResponse]);
        setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100 dark:bg-gray-900">
      <Card className="w-full max-w-2xl h-[90vh] flex flex-col shadow-lg">
        <CardHeader className="flex flex-row items-center justify-between border-b p-4">
          <div className="flex items-center gap-3">
            <Bot className="w-8 h-8 text-primary" />
            <div>
              <CardTitle className="text-lg">{modelName}</CardTitle>
              <p className="text-xs text-muted-foreground">ИИ-ассистент на базе WordWise.js</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                         <Button variant="ghost" size="icon" aria-label="Toggle Predictions" onClick={() => setShowPredictions(p => !p)}>
                            {showPredictions ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
                         </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                        <p>{showPredictions ? 'Скрыть монитор предсказаний' : 'Показать монитор предсказаний'}</p>
                    </TooltipContent>
                </Tooltip>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Link href="/wordwise" passHref>
                          <Button variant="ghost" size="icon" aria-label="WordWise.js Training">
                            <BrainCircuit className="w-5 h-5" />
                          </Button>
                        </Link>
                    </TooltipTrigger>
                    <TooltipContent>
                        <p>Перейти в тренажерный зал</p>
                    </TooltipContent>
                </Tooltip>
            </TooltipProvider>
          </div>
        </CardHeader>
        <CardContent className="flex-grow p-0">
          <ScrollArea className="h-full" ref={scrollAreaRef}>
             <div className="p-4 space-y-4">
               {messages.length === 0 && (
                 <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                    <p>Сообщений пока нет.</p>
                    <p className="text-sm">Чтобы начать, отправьте сообщение или обучите модель в разделе <BrainCircuit className="inline h-4 w-4" />.</p>
                 </div>
               )}
              {messages.map((message, index) => (
                <div key={index} className={`flex items-start gap-3 ${message.role === 'user' ? 'justify-end' : ''}`}>
                  {message.role === 'assistant' && (
                    <Avatar className="w-8 h-8">
                      <AvatarFallback>AI</AvatarFallback>
                    </Avatar>
                  )}
                  <div className={`rounded-lg px-3 py-2 max-w-[80%] ${message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                    }`}>
                    <p className="text-sm whitespace-pre-wrap">{message.content || '...'}</p>
                  </div>
                  {message.role === 'user' && (
                    <Avatar className="w-8 h-8">
                      <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}
              {isLoading && messages[messages.length-1]?.role === 'assistant' && (
                  <div className="flex items-start gap-3">
                     <Avatar className="w-8 h-8">
                       <AvatarFallback>AI</AvatarFallback>
                     </Avatar>
                     <div className="rounded-lg px-3 py-2 bg-muted">
                        <p className="text-sm">...</p>
                     </div>
                  </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
        <div className="p-4 border-t bg-background">
          {showPredictions && <PredictionVisualizer predictions={latestPredictions} />}
          <form onSubmit={handleSubmit} className="flex gap-2 mt-2">
            <Input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Спросите что-нибудь..."
              autoComplete="off"
              disabled={isLoading}
              className="flex-grow"
            />
            <Button type="submit" disabled={isLoading}>
              Отправить
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
}
