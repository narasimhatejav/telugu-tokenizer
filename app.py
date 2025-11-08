#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Gradio App for Telugu BPE Tokenizer
Interactive interface to tokenize Telugu text
"""

import gradio as gr
import json
import pickle
from train_tokenizer import encode, decode

# Load tokenizer at startup
print("Loading tokenizer...")
try:
    with open('tokenizer_vocab.json', 'r', encoding='utf-8') as f:
        vocab_json = json.load(f)
        VOCAB = {int(k): bytes(v) for k, v in vocab_json.items()}

    with open('tokenizer_merges.pkl', 'rb') as f:
        MERGES = pickle.load(f)

    VOCAB_SIZE = len(VOCAB)
    print(f"Tokenizer loaded successfully! Vocabulary size: {VOCAB_SIZE}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    VOCAB = None
    MERGES = None
    VOCAB_SIZE = 0

# Color palette for tokens
COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#FFB3E6", "#E6B3FF", "#FFC9BA", "#C9FFE6", "#E6FFC9",
    "#FFE6B3", "#B3E6FF", "#FFB3D9", "#D9FFB3", "#B3D9FF",
    "#FFCCE6", "#E6FFCC", "#CCFFE6", "#E6CCFF", "#FFCCD9"
]


def tokenize_text(text, show_whitespace=False):
    """
    Tokenize input text and return colored visualization and token IDs

    Args:
        text: Input text to tokenize
        show_whitespace: Whether to show spaces as ␣

    Returns:
        Tuple of (token_count, colored_html, token_ids_string)
    """
    if not text.strip():
        return "0", "", ""

    if VOCAB is None or MERGES is None:
        return "0", "Error: Tokenizer not loaded properly.", ""

    # Encode text
    tokens = encode(text, MERGES)
    token_count = len(tokens)

    # Create colored HTML output
    # Since BPE tokens can split UTF-8 sequences, we need to decode all bytes together
    # then figure out which characters each token contributes
    html_parts = ['<div style="line-height: 2.5; font-size: 20px; font-family: \'Noto Sans Telugu\', sans-serif; display: flex; flex-wrap: wrap; gap: 2px;">']

    # Concatenate all token bytes
    all_bytes = b''.join([VOCAB[tid] for tid in tokens])
    full_decoded = all_bytes.decode('utf-8')

    # For each token, figure out which characters it contributes
    byte_pos = 0
    for i, token_id in enumerate(tokens):
        token_bytes = VOCAB[token_id]
        token_len = len(token_bytes)

        # Get bytes up to this point
        bytes_before = all_bytes[:byte_pos]
        bytes_including = all_bytes[:byte_pos + token_len]

        # Decode to find character boundaries
        # Use 'ignore' to handle partial sequences
        chars_before = len(bytes_before.decode('utf-8', errors='ignore'))
        chars_including = len(bytes_including.decode('utf-8', errors='ignore'))

        # Extract the characters this token contributes
        token_text = full_decoded[chars_before:chars_including]

        # Escape HTML special characters
        token_text = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Show spaces visibly if enabled
        if show_whitespace:
            token_text = token_text.replace(' ', '␣')

        color = COLORS[i % len(COLORS)]
        html_parts.append(
            f'<span class="token" data-token-id="{i}" '
            f'style="background-color: {color}; padding: 4px 8px; '
            f'border-radius: 4px; cursor: pointer; display: inline-block;">'
            f'{token_text}</span>'
        )

        byte_pos += token_len

    html_parts.append('</div>')

    # Add CSS and JavaScript for hover functionality
    # Use a unique ID for each render to avoid conflicts
    import random
    unique_id = f"tokenizer_{random.randint(1000, 9999)}"

    html_output = f'''
<div id="{unique_id}">
<style>
#{unique_id} .token {{
    transition: all 0.2s;
    color: inherit;
}}
#{unique_id} .token:hover {{
    transform: scale(1.1);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
#{unique_id} .token.highlight-token {{
    background-color: white !important;
    color: black !important;
    transform: scale(1.1);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
#{unique_id} .token-id-num {{
    transition: all 0.2s;
    font-weight: 500;
}}
#{unique_id} .token-id-num.highlight {{
    font-weight: bold;
    text-decoration: underline;
    transform: scale(1.1);
}}
</style>
''' + ''.join(html_parts) + '''
<script>
(function() {
    const containerId = "''' + unique_id + '''";
    const container = document.getElementById(containerId);
    if (!container) return;

    const tokens = container.querySelectorAll('.token');
    const tokenIdNums = container.querySelectorAll('.token-id-num');

    tokens.forEach(token => {
        token.addEventListener('mouseenter', function() {
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokenIdNums[tokenIndex]) {
                tokenIdNums[tokenIndex].classList.add('highlight');
            }
        });

        token.addEventListener('mouseleave', function() {
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokenIdNums[tokenIndex]) {
                tokenIdNums[tokenIndex].classList.remove('highlight');
            }
        });
    });

    tokenIdNums.forEach(tokenIdNum => {
        tokenIdNum.addEventListener('mouseenter', function() {
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokens[tokenIndex]) {
                tokens[tokenIndex].classList.add('highlight-token');
            }
        });

        tokenIdNum.addEventListener('mouseleave', function() {
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokens[tokenIndex]) {
                tokens[tokenIndex].classList.remove('highlight-token');
            }
        });
    });
})();
</script>
</div>
'''

    # Create token IDs output as comma-separated list
    # Must use same unique_id to allow cross-interaction
    token_id_list = []
    for i, token_id in enumerate(tokens):
        token_id_list.append(
            f'<span class="token-id-num" data-token-id="{i}" style="cursor: pointer;">{token_id}</span>'
        )

    token_ids_html = f'''
<div id="{unique_id}_ids">
<style>
#{unique_id}_ids .token-id-num {{
    transition: all 0.2s;
    font-weight: 500;
    cursor: pointer;
}}
#{unique_id}_ids .token-id-num.highlight {{
    font-weight: bold;
    text-decoration: underline;
    transform: scale(1.1);
}}
</style>
<div style="font-size: 16px; font-family: monospace; line-height: 1.8;">
{', '.join(token_id_list)}
</div>
<script>
(function() {{
    const mainContainerId = "{unique_id}";
    const idsContainerId = "{unique_id}_ids";

    const mainContainer = document.getElementById(mainContainerId);
    const idsContainer = document.getElementById(idsContainerId);

    if (!mainContainer || !idsContainer) {{
        console.log("Containers not found, retrying...");
        setTimeout(arguments.callee, 100);
        return;
    }}

    const tokens = mainContainer.querySelectorAll('.token');
    const tokenIdNums = idsContainer.querySelectorAll('.token-id-num');

    console.log("Found tokens:", tokens.length, "Found IDs:", tokenIdNums.length);

    tokens.forEach(token => {{
        token.addEventListener('mouseenter', function() {{
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokenIdNums[tokenIndex]) {{
                tokenIdNums[tokenIndex].classList.add('highlight');
            }}
        }});

        token.addEventListener('mouseleave', function() {{
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokenIdNums[tokenIndex]) {{
                tokenIdNums[tokenIndex].classList.remove('highlight');
            }}
        }});
    }});

    tokenIdNums.forEach(tokenIdNum => {{
        tokenIdNum.addEventListener('mouseenter', function() {{
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokens[tokenIndex]) {{
                tokens[tokenIndex].classList.add('highlight-token');
            }}
        }});

        tokenIdNum.addEventListener('mouseleave', function() {{
            const tokenIndex = parseInt(this.getAttribute('data-token-id'));
            if (tokens[tokenIndex]) {{
                tokens[tokenIndex].classList.remove('highlight-token');
            }}
        }});
    }});
}})();
</script>
</div>
'''

    return str(token_count), html_output, token_ids_html


# Example Telugu texts
EXAMPLES = [
    ["తెలుగు", False],
    ["నమస్కారం", False],
    ["తెలుగు భాష చాలా అందంగా ఉంది", False],
    ["హైదరాబాద్ భారతదేశంలో ఒక ముఖ్యమైన నగరం", False],
    ["మనం తెలుగులో మాట్లాడుతున్నాము", False],
]

# Create Gradio interface with forced light mode
theme = gr.themes.Default().set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_50",
)

with gr.Blocks(
    title="Telugu BPE Tokenizer",
    theme=theme,
    css="""
        body, .gradio-container {
            color-scheme: light !important;
        }
        .dark, [data-theme="dark"] {
            color-scheme: light !important;
        }
    """,
    js="() => { document.body.classList.remove('dark'); }"
) as demo:
    gr.Markdown(
        """
        # 🔤 Telugu BPE Tokenizer

        This is a **Byte Pair Encoding (BPE)** tokenizer trained on Telugu Wikipedia articles.

        ### 📈 Model Statistics:
        - **Vocabulary Size**: {VOCAB_SIZE:,} tokens
        - **Base Tokens**: 256 (UTF-8 bytes)
        - **Learned Merges**: {MERGES_COUNT:,}
        - **Training Corpus**: 300 Telugu Wikipedia articles

        ### 🎯 Try it out!
        Enter any Telugu text below to see how it gets tokenized. **Hover over the colored tokens to see the corresponding token IDs!**
        """.format(
            VOCAB_SIZE=VOCAB_SIZE,
            MERGES_COUNT=len(MERGES) if MERGES else 0
        )
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Telugu Text",
                placeholder="తెలుగు టెక్స్ట్ ఇక్కడ టైప్ చేయండి...",
                lines=10
            )

        with gr.Column(scale=1):
            token_count_output = gr.Textbox(
                label="Token count",
                interactive=False,
                lines=1
            )
            colored_output = gr.HTML(label="Tokenized Text (Colored)")
            token_ids_output = gr.HTML(label="Token IDs")
            show_whitespace = gr.Checkbox(label="Show whitespace", value=False)

    # Examples section
    gr.Markdown("### 📚 Example Texts:")
    gr.Examples(
        examples=[["తెలుగు"], ["నమస్కారం"], ["తెలుగు భాష చాలా అందంగా ఉంది"], ["హైదరాబాద్ భారతదేశంలో ఒక ముఖ్యమైన నగరం"], ["మనం తెలుగులో మాట్లాడుతున్నాము"]],
        inputs=input_text,
        cache_examples=False
    )

    # Connect input changes to tokenize function
    input_text.change(
        fn=tokenize_text,
        inputs=[input_text, show_whitespace],
        outputs=[token_count_output, colored_output, token_ids_output]
    )

    # Also trigger when whitespace checkbox changes
    show_whitespace.change(
        fn=tokenize_text,
        inputs=[input_text, show_whitespace],
        outputs=[token_count_output, colored_output, token_ids_output]
    )

    # Footer
    gr.Markdown(
        """
        ---
        ### ℹ️ About BPE
        Byte Pair Encoding is a subword tokenization algorithm that:
        1. Starts with 256 base tokens (UTF-8 bytes)
        2. Iteratively merges the most frequent byte pairs
        3. Builds a vocabulary that efficiently represents the language

        This tokenizer achieves excellent compression for Telugu text while maintaining perfect reconstruction.

        ### 💡 How to use:
        - Each token is shown with a different color
        - Hover over a colored token to highlight its corresponding token ID
        - Hover over a token ID to highlight the corresponding text token
        """
    )


if __name__ == "__main__":
    demo.launch(share=True)
