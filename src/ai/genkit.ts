'use server';

import {genkit} from 'genkit';
import {defineModel, GenerationCommonConfigSchema} from '@genkit-ai/ai';
import {z} from 'zod';

const myModel = defineModel(
  {
    name: 'my-custom-model',
    configSchema: GenerationCommonConfigSchema,
    info: {
      label: 'My Custom Model',
      supports: {
        media: false,
        output: ['text'],
        input: ['text'],
      },
    },
  },
  async (request, stream) => {
    await new Promise(resolve => setTimeout(resolve, 500));
    const responseText = `This is a placeholder response for: ${request.prompt[0].text}`;
    if (stream) {
      stream.text(responseText);
      return {
        candidates: [
          {
            index: 0,
            finishReason: 'stop',
            output: {
                role: 'model',
                content: [{text: responseText}]
            }
          },
        ],
      };
    } else {
      return {
        candidates: [
          {
            index: 0,
            finishReason: 'stop',
            output: {
                role: 'model',
                content: [{text: responseText}]
            }
          },
        ],
      };
    }
  }
);

export const ai = genkit({
  plugins: [],
  models: [myModel],
});
