import streamlit as st
import torch
import random
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from transformers import pipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Configure Streamlit page
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-text {
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session states
if 'model' not in st.session_state:
    st.session_state.model = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = None

# Title and description
st.title("🎨 AI Image Generator")
st.markdown("""
    Transform your imagination into visual art using AI.
    Simply enter a description and watch the magic happen!
""")

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

@st.cache_resource
def initialize_models():
    """Initialize all required models"""
    try:
        # Set up device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize sentiment analyzer
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english"
        )
        
        # Initialize topic classifier
        topic_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Initialize Stable Diffusion
        model_id = "Lykon/dreamshaper-8"
        stable_diffusion = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        # Set up scheduler
        stable_diffusion.scheduler = DPMSolverMultistepScheduler.from_config(
            stable_diffusion.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2
        )
        
        if device == "cuda":
            stable_diffusion.enable_xformers_memory_efficient_attention()
        
        stable_diffusion = stable_diffusion.to(device)
        
        return device, sentiment_analyzer, topic_classifier, stable_diffusion
    
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None, None, None

def analyze_prompt(prompt, sentiment_analyzer, topic_classifier):
    """Analyze the input prompt"""
    try:
        # Sentiment analysis
        sentiment_result = sentiment_analyzer(prompt)[0]
        
        # Topic classification
        candidate_topics = [
            "landscape", "portrait", "abstract", "nature", "urban",
            "fantasy", "sci-fi", "realistic", "artistic", "architectural"
        ]
        topic_result = topic_classifier(prompt, candidate_topics)
        
        # NLTK analysis
        tokens = word_tokenize(prompt)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        subjects = [word for word, tag in pos_tags if tag.startswith(('NN', 'JJ'))]
        
        # Style analysis
        style_keywords = {
            'realistic': ['realistic', 'photographic', 'natural', 'real'],
            'artistic': ['artistic', 'painted', 'stylized', 'abstract'],
            'fantasy': ['magical', 'fantasy', 'mythical', 'mystical'],
            'sci-fi': ['futuristic', 'sci-fi', 'technological', 'cyber']
        }
        
        style_scores = {
            style: sum(1 for keyword in keywords if keyword in prompt.lower())
            for style, keywords in style_keywords.items()
        }
        primary_style = max(style_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'sentiment': sentiment_result['label'],
            'sentiment_score': sentiment_result['score'],
            'primary_topic': topic_result['labels'][0],
            'topic_score': topic_result['scores'][0],
            'key_elements': subjects,
            'style': primary_style,
            'named_entities': str(named_entities)
        }
    
    except Exception as e:
        st.error(f"Error analyzing prompt: {e}")
        return None

def generate_enhanced_prompt(analysis):
    """Generate an enhanced prompt based on analysis"""
    base_modifiers = {
        'realistic': [
            "highly detailed", "professional photography", "8k uhd",
            "sharp focus", "dramatic lighting", "photorealistic"
        ],
        'artistic': [
            "digital art", "concept art", "trending on artstation",
            "artistic", "stylized", "award winning"
        ],
        'fantasy': [
            "magical", "ethereal", "mystical atmosphere", "fantasy art",
            "dreamlike", "surreal"
        ],
        'sci-fi': [
            "futuristic", "sci-fi", "cyberpunk", "technological",
            "neon lighting", "ultramodern"
        ]
    }
    
    style_modifiers = base_modifiers.get(analysis['style'], base_modifiers['realistic'])
    
    quality_modifiers = [
        "masterpiece", "high quality", "detailed",
        "professional", "cinematic composition"
    ]
    
    atmosphere_modifiers = (
        ["vibrant", "beautiful", "stunning", "epic"]
        if analysis['sentiment'] == 'POSITIVE'
        else ["moody", "dramatic", "mysterious", "intense"]
    )
    
    selected_modifiers = (
        random.sample(style_modifiers, 2) +
        random.sample(quality_modifiers, 2) +
        random.sample(atmosphere_modifiers, 1)
    )
    
    return f"{', '.join(selected_modifiers)}"

def generate_images(model, prompt, prompt_analysis, device, num_images=1):
    """Generate images based on the prompt"""
    try:
        prompt_modifiers = generate_enhanced_prompt(prompt_analysis)
        full_prompt = f"{prompt}, {prompt_modifiers}"
        
        st.write("🎨 Using enhanced prompt:", full_prompt)
        
        negative_prompt = (
            "blur, watermark, text, logo, low quality, deformed, bad anatomy, "
            "disfigured, poorly drawn, bad proportions, duplicate, morbid, "
            "out of frame, extra fingers, mutated hands, monochrome, grainy"
        )
        
        inference_steps = 75 if prompt_analysis['style'] in ['realistic', 'artistic'] else 50
        guidance_scale = 8.5 if prompt_analysis['style'] in ['fantasy', 'sci-fi'] else 7.5
        
        generated_images = []
        seeds = random.sample(range(1, 1000000), num_images)
        
        for seed in seeds:
            with st.spinner(f'Generating image {len(generated_images) + 1} of {num_images}...'):
                generator = torch.Generator(device).manual_seed(seed)
                
                image = model(
                    full_prompt,
                    height=768,
                    width=768,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    generator=generator
                ).images[0]
                
                generated_images.append((image, seed, full_prompt))
        
        return generated_images
    
    except Exception as e:
        st.error(f"Error generating images: {e}")
        return None

def main():
    # Download NLTK data
    download_nltk_data()
    
    # Initialize models
    device, sentiment_analyzer, topic_classifier, model = initialize_models()
    
    # Create sidebar for settings
    st.sidebar.title("⚙️ Settings")
    num_images = st.sidebar.slider("Number of images to generate:", 1, 4, 1)
    
    # Create input form
    with st.form("generation_form"):
        prompt = st.text_area(
            "Enter your description:",
            placeholder="Example: A serene mountain landscape at sunset with a crystal clear lake...",
            height=100
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            generate_button = st.form_submit_button("🎨 Generate Images")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear Results")
    
    # Handle clear button
    if clear_button:
        st.session_state.generated_images = None
        st.session_state.analysis = None
        st.experimental_rerun()
    
    # Handle generation
    if generate_button and prompt:
        if not model:
            st.error("Models failed to initialize. Please refresh the page.")
            return
        
        with st.spinner("Analyzing prompt..."):
            analysis = analyze_prompt(prompt, sentiment_analyzer, topic_classifier)
            if analysis:
                st.session_state.analysis = analysis
                
                with st.spinner("Generating images..."):
                    generated_images = generate_images(
                        model, prompt, analysis, device, num_images
                    )
                    if generated_images:
                        st.session_state.generated_images = generated_images
    
    # Display results
    if st.session_state.generated_images and st.session_state.analysis:
        st.markdown("---")
        st.subheader("🖼️ Generated Images")
        
        # Create columns for images
        cols = st.columns(num_images)
        
        # Display images
        for idx, (image, seed, full_prompt) in enumerate(st.session_state.generated_images):
            with cols[idx]:
                st.image(image, use_column_width=True)
                st.caption(f"Seed: {seed}")
        
        # Display analysis
        st.markdown("---")
        st.subheader("📊 Prompt Analysis")
        
        analysis = st.session_state.analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Style and Topic")
            st.write(f"🎨 **Style:** {analysis['style']}")
            st.write(f"📌 **Primary Topic:** {analysis['primary_topic']}")
            st.write(f"🎭 **Sentiment:** {analysis['sentiment']}")
        
        with col2:
            st.markdown("##### Key Elements")
            st.write("🔑 **Detected Elements:**")
            st.write(", ".join(analysis['key_elements']))

if __name__ == "__main__":
    main()