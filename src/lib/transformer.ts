// src/lib/transformer.ts
import { Tensor } from './tensor';
import { Layer, Linear, LayerNorm, softmax } from './layers';

class MultiHeadAttention extends Layer {
    wq: Linear;
    wk: Linear;
    wv: Linear;
    wo: Linear;
    numHeads: number;
    dModel: number;
    dk: number;
    parameters: Tensor[];

    constructor(dModel: number, numHeads: number) {
        super();
        if (dModel % numHeads !== 0) {
            throw new Error("dModel must be divisible by numHeads");
        }
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dk = dModel / numHeads;

        this.wq = new Linear(dModel, dModel);
        this.wk = new Linear(dModel, dModel);
        this.wv = new Linear(dModel, dModel);
        this.wo = new Linear(dModel, dModel);
        
        this.wq.weights.name = 'wq_weights'; this.wq.bias.name = 'wq_bias';
        this.wk.weights.name = 'wk_weights'; this.wk.bias.name = 'wk_bias';
        this.wv.weights.name = 'wv_weights'; this.wv.bias.name = 'wv_bias';
        this.wo.weights.name = 'wo_weights'; this.wo.bias.name = 'wo_bias';

        this.parameters = [
            ...this.wq.getParameters(),
            ...this.wk.getParameters(),
            ...this.wv.getParameters(),
            ...this.wo.getParameters()
        ];
    }

    private splitHeads(x: Tensor): Tensor {
        const [batchSize, seqLen, dModel] = x.shape;
        // Reshape to [B, S, H, Dk] then transpose to [B, H, S, Dk]
        return x.reshape([batchSize, seqLen, this.numHeads, this.dk]).transpose(1, 2);
    }

    private combineHeads(x: Tensor): Tensor {
        // x is [B, H, S, Dk]
        const [batchSize, numHeads, seqLen, dk] = x.shape;
        // Transpose to [B, S, H, Dk] then reshape to [B, S, Dm]
        return x.transpose(1, 2).reshape([batchSize, seqLen, this.dModel]);
    }

    forward(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | null = null): Tensor {
        const q_proj = this.wq.forward(q); // [B, S, Dm]
        const k_proj = this.wk.forward(k); // [B, S, Dm]
        const v_proj = this.wv.forward(v); // [B, S, Dm]

        const q_split = this.splitHeads(q_proj); // [B, H, S, Dk]
        const k_split = this.splitHeads(k_proj); // [B, H, S, Dk]
        const v_split = this.splitHeads(v_proj); // [B, H, S, Dk]

        // Scaled Dot-Product Attention
        const scores = q_split.dot(k_split.transpose(-2, -1)); // [B, H, S, S]
        const scaledScores = scores.divScalar(Math.sqrt(this.dk));

        if (mask) {
            // Add a very small number to masked positions before softmax
            scaledScores.add(mask, true); 
        }

        const attentionWeights = softmax(scaledScores); // [B, H, S, S]
        const context = attentionWeights.dot(v_split); // [B, H, S, Dk]

        const combined = this.combineHeads(context); // [B, S, Dm]
        const output = this.wo.forward(combined); // [B, S, Dm]

        return output;
    }
}


class FeedForward extends Layer {
    linear1: Linear;
    linear2: Linear;
    parameters: Tensor[];

    constructor(dModel: number, dff: number) {
        super();
        this.linear1 = new Linear(dModel, dff);
        this.linear2 = new Linear(dff, dModel);
        this.linear1.weights.name = 'ffn_w1'; this.linear1.bias.name = 'ffn_b1';
        this.linear2.weights.name = 'ffn_w2'; this.linear2.bias.name = 'ffn_b2';
        this.parameters = [...this.linear1.getParameters(), ...this.linear2.getParameters()];
    }

    forward(x: Tensor): Tensor {
        let y = this.linear1.forward(x);
        y = y.apply(x => Math.max(0, x), (x, y, g) => (x > 0 ? g : 0)); // ReLU
        y = this.linear2.forward(y);
        return y;
    }
}


export class TransformerEncoderBlock extends Layer {
    mha: MultiHeadAttention;
    ffn: FeedForward;
    layernorm1: LayerNorm;
    layernorm2: LayerNorm;
    parameters: Tensor[];

    constructor(dModel: number, numHeads: number, dff: number) {
        super();
        this.mha = new MultiHeadAttention(dModel, numHeads);
        this.ffn = new FeedForward(dModel, dff);
        this.layernorm1 = new LayerNorm(dModel);
        this.layernorm2 = new LayerNorm(dModel);
        this.layernorm1.gamma.name = 'ln1_gamma'; this.layernorm1.beta.name = 'ln1_beta';
        this.layernorm2.gamma.name = 'ln2_gamma'; this.layernorm2.beta.name = 'ln2_beta';
        
        this.parameters = [
            ...this.mha.getParameters(),
            ...this.ffn.getParameters(),
            ...this.layernorm1.getParameters(),
            ...this.layernorm2.getParameters()
        ];
    }

    forward(x: Tensor): Tensor {
        // Attention sublayer
        const attnOutput = this.mha.forward(x, x, x);
        const out1 = this.layernorm1.forward(x.add(attnOutput));

        // Feed-forward sublayer
        const ffnOutput = this.ffn.forward(out1);
        const out2 = this.layernorm2.forward(out1.add(ffnOutput));
        
        return out2;
    }
}
