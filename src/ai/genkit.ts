import {genkit, Plugin} from 'genkit';
import {googleAI} from '@genkit-ai/google-ai';
import {firebase} from '@genkit-ai/firebase';
import {genkitEval} from 'genkit-eval';

const plugins: Plugin<any>[] = [googleAI(), firebase(), genkitEval()];

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
