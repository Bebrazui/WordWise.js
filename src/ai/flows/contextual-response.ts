'use server';
/**
 * @fileOverview This file implements the AI flow for generating contextual chat responses.
 *
 * - generateChatResponse: The main function to generate a response.
 * - ChatMessage: The type definition for a chat message.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

// Define the schema for a single chat message
const ChatMessageSchema = z.object({
  role: z.enum(['user', 'assistant']),
  content: z.string(),
});
export type ChatMessage = z.infer<typeof ChatMessageSchema>;

// Define the input schema for the main flow
const ChatInputSchema = z.object({
  message: z.string(),
  history: z.array(ChatMessageSchema),
  isExperimental: z.boolean(),
});

/**
 * Generates a chat response using a generative model.
 * @param input The user's latest message.
 * @param history A history of the conversation.
 * @param isExperimental A flag to switch between different model behaviors.
 * @returns A promise that resolves to the bot's string response.
 */
export async function generateChatResponse(
  input: string,
  history: ChatMessage[],
  isExperimental: boolean
): Promise<string> {
  // Use the underlying Genkit flow
  const response = await contextualResponseFlow({
    message: input,
    history,
    isExperimental,
  });
  return response;
}

// Define the main Genkit flow for generating contextual responses
const contextualResponseFlow = ai.defineFlow(
  {
    name: 'contextualResponseFlow',
    inputSchema: ChatInputSchema,
    outputSchema: z.string(),
  },
  async ({message, history, isExperimental}) => {
    // Determine the model and system prompt based on the mode
    const model = isExperimental
      ? 'googleai/gemini-1.5-flash-preview'
      : 'googleai/gemini-1.5-flash-preview';
    
    const systemPrompt = isExperimental
      ? `You are Bot Q 0.3 (Generative). You are a helpful and creative AI assistant. You can write code and generate creative text.`
      : `You are Bot Q 0.2 (Quantum). You are a helpful assistant. Provide concise and accurate answers.`;

    const {output} = await ai.generate({
      model,
      system: systemPrompt,
      history: history.map(msg => ({
        role: msg.role,
        content: [{ text: msg.content }],
      })),
      prompt: message,
      config: {
        // Lower temperature for the standard model for more predictable answers
        // Higher temperature for the experimental model for more creativity
        temperature: isExperimental ? 0.7 : 0.3,
      },
    });

    return output ?? "I'm sorry, I couldn't generate a response.";
  }
);
