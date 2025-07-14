
"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";

import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Code, Bot } from "lucide-react";
import { cn } from "@/lib/utils";

const formSchema = z.object({
  prompt: z.string().min(1, "Prompt cannot be empty"),
});

export function CodeAssistant() {
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      prompt: "",
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setIsLoading(true);
    setGeneratedCode("");

    // TODO: Implement the call to the AI flow
    // For now, we'll just simulate a delay and return a placeholder
    await new Promise((resolve) => setTimeout(resolve, 1000));
    
    const placeholderCode = `// AI-generated code for: "${values.prompt}"
function helloWorld() {
  console.log("Hello, world!");
}`;
    
    setGeneratedCode(placeholderCode);
    setIsLoading(false);
  }

  return (
    <div className="flex flex-col h-screen bg-background text-foreground p-4 md:p-8">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Code className="h-6 w-6" />
          Code Assistant
        </h1>
        <Button variant="outline" asChild>
          <a href="/">Back to Chat</a>
        </Button>
      </header>
      <main className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6 overflow-hidden">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5" />
              Your Request
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="space-y-4"
              >
                <FormField
                  control={form.control}
                  name="prompt"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>
                        <Textarea
                          placeholder="e.g., 'Create a function to add two numbers'"
                          className="min-h-[200px] text-base"
                          disabled={isLoading}
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <Button type="submit" disabled={isLoading} className="w-full">
                  {isLoading ? "Generating..." : "Generate Code"}
                </Button>
              </form>
            </CardContent>
          </Card>
        </Card>
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Generated Code</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden">
            <ScrollArea className="h-full">
              <pre className="bg-muted p-4 rounded-md text-sm text-muted-foreground whitespace-pre-wrap">
                {generatedCode || "Code will appear here..."}
              </pre>
            </ScrollArea>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
