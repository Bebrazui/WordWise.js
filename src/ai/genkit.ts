import {genkit, Plugin} from 'genkit';
import {googleAI} from '@genkit-ai/googleai';
import {firebase} from '@genkit-ai/firebase';

const plugins: Plugin<any>[] = [googleAI(), firebase()];

if (process.env.GENKIT_ENV === 'dev') {
  const {dotprompt} = await import('genkitx-dotprompt');
  plugins.push(dotprompt());
}

export const ai = genkit({
  plugins,
  flowStateStore: 'firebase',
  traceStore: 'firebase',
  enableTracingAndMetrics: true,
});
