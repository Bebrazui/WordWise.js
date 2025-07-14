
"use client";

import { useState, useRef, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { contextualResponse } from "@/ai/flows/contextual-response";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Send, Bot, User, BrainCircuit, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';


type Message = {
  id: number;
  text: string;
  sender: "user" | "ai";
  isTyping?: boolean;
};

const formSchema = z.object({
  message: z.string().min(1, "Message cannot be empty"),
});

const initialMessage: Message = {
  id: 1,
  text: "Привет! Я WordWise, ваш личный помощник. Спроси меня о чем-нибудь или попроси написать код.",
  sender: "ai",
};

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([initialMessage]);
  const [isLoading, setIsLoading] = useState(false);
  const [experimental, setExperimental] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      message: "",
    },
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  async function onSubmit(values: z.infer<typeof formSchema>) {
    const userInput = values.message;

    const userMessage: Message = {
      id: Date.now(),
      text: userInput,
      sender: "user",
    };

    const typingMessage: Message = {
      id: Date.now() + 1,
      text: "",
      sender: "ai",
      isTyping: true,
    };

    const newMessages = [...messages, userMessage];
    setMessages([...newMessages, typingMessage]);
    form.reset();
    setIsLoading(true);

    try {
      const history = newMessages
        .filter((m) => m.id !== initialMessage.id)
        .slice(-4, -1)
        .map((m) => m.text);


      const response = await contextualResponse({
        userInput: userInput,
        history,
        experimental,
      });

      const aiMessage: Message = {
        id: Date.now() + 2,
        text: response.aiResponse,
        sender: "ai",
      };
      setMessages((prev) => [...prev.filter((m) => !m.isTyping), aiMessage]);
    } catch (error) {
      console.error("Error getting AI response:", error);
      toast({
        variant: "destructive",
        title: "Ой, что-то пошло не так.",
        description:
          "Возникла проблема с получением ответа. Пожалуйста, попробуйте еще раз.",
      });
      setMessages((prev) => prev.filter((m) => !m.isTyping));
    } finally {
      setIsLoading(false);
    }
  }
  
  const CodeBlock = ({
    node,
    inline,
    className,
    children,
    ...props
  }: any) => {
    const match = /language-(\w+)/.exec(className || '');
    // vscDarkPlus is a good dark theme for code.
    // We can also use vs (light), coy, okaidia, etc.
    return !inline && match ? (
      <SyntaxHighlighter
        style={vscDarkPlus}
        language={match[1]}
        PreTag="div"
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    ) : (
      <code className={className} {...props}>
        {children}
      </code>
    );
  };


  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <header className="flex items-center justify-between p-4 border-b shadow-sm bg-card shrink-0">
        <h1 className="text-xl font-bold text-card-foreground">
          WordWise Chat
        </h1>
        <div className="flex items-center gap-2">
          <Button
            variant={"secondary"}
            size="sm"
            className="gap-2 pointer-events-none"
          >
            <BrainCircuit className="h-4 w-4" />
            {experimental ? "Bot Q 0.3 (TensorFlow.js)" : "Bot Q 0.2 (Quantum)"}
          </Button>
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <Settings className="h-5 w-5" />
                <span className="sr-only">Настройки</span>
              </Button>
            </SheetTrigger>
            <SheetContent>
              <SheetHeader>
                <SheetTitle>Настройки</SheetTitle>
                <SheetDescription>
                  Здесь вы можете управлять поведением чат-бота.
                </SheetDescription>
              </SheetHeader>
              <div className="grid gap-4 py-4">
                <div className="flex items-center justify-between space-x-2 p-2 rounded-lg border">
                  <Label htmlFor="experimental-mode" className="flex flex-col space-y-1">
                    <span>Экспериментальный режим</span>
                    <span className="font-normal leading-snug text-muted-foreground">
                      Использовать легковесный TF.js модель для более быстрых, но менее подробных ответов.
                    </span>
                  </Label>
                  <Switch
                    id="experimental-mode"
                    checked={experimental}
                    onCheckedChange={setExperimental}
                  />
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-4">
          <div className="space-y-6 pr-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex items-start gap-3 animate-in fade-in-0 slide-in-from-bottom-4 duration-500",
                  message.sender === "user" ? "justify-end" : "justify-start"
                )}
              >
                {message.sender === "ai" && (
                  <Avatar className="h-9 w-9 border shrink-0">
                    <AvatarFallback className="bg-primary text-primary-foreground">
                      <Bot className="h-5 w-5" />
                    </AvatarFallback>
                  </Avatar>
                )}
                <div
                  className={cn(
                    "flex flex-col gap-1 w-full max-w-md lg:max-w-2xl",
                    message.sender === "user" ? "items-end" : "items-start"
                  )}
                >
                  {message.isTyping ? (
                    <div className="bg-card p-3 rounded-lg flex items-center space-x-1.5 shadow-md">
                      <span className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                      <span className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                      <span className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce"></span>
                    </div>
                  ) : (
                    <div
                      className={cn(
                        "p-3 rounded-lg shadow-md",
                        message.sender === "user"
                          ? "bg-accent text-accent-foreground rounded-br-none"
                          : "bg-card text-card-foreground rounded-bl-none"
                      )}
                    >
                      <ReactMarkdown
                        components={{
                            code: CodeBlock,
                        }}
                        className="prose prose-sm dark:prose-invert max-w-none"
                      >
                        {message.text}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
                {message.sender === "user" && (
                  <Avatar className="h-9 w-9 border shrink-0">
                    <AvatarFallback>
                      <User className="h-5 w-5" />
                    </AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
      </main>
      <footer className="p-4 border-t bg-card shrink-0">
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(onSubmit)}
            className="flex items-center gap-2"
          >
            <FormField
              control={form.control}
              name="message"
              render={({ field }) => (
                <FormItem className="flex-1">
                  <FormControl>
                    <Input
                      placeholder="Type a message or ask to write code..."
                      autoComplete="off"
                      disabled={isLoading}
                      {...field}
                      className="text-base"
                    />
                  </FormControl>
                </FormItem>
              )}
            />
            <Button type="submit" size="icon" disabled={isLoading}>
              <Send className="h-5 w-5" />
              <span className="sr-only">Send</span>
            </Button>
          </form>
        </Form>
      </footer>
    </div>
  );
}
