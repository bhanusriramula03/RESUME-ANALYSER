import streamlit as st
import PyPDF2
import re
import io
import plotly.graph_objects as go
import nltk
# No longer using word_tokenize, using regex instead
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import math

# Ensure required NLTK data is available
def download_nltk_data():
    """Download required NLTK resources if not already available."""
    # Download all required resources without checking first
    # This is more reliable on Streamlit Cloud
    with st.spinner('Downloading required language data...'):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        # We'll use a regex-based tokenizer instead of punkt to avoid the punkt_tab error

# Text processing functions
def process_text(text):
    """
    Process text by tokenizing, lemmatizing, and removing stopwords.
    
    Args:
        text (str): Raw text to process
        
    Returns:
        list: Processed tokens
    """
    # Initialize tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Process text using regex-based tokenization instead of word_tokenize
    # This avoids the punkt_tab dependency issue on Streamlit Cloud
    text = text.lower()
    # Simple regex to split on non-alphanumeric characters
    raw_tokens = re.findall(r'\w+', text)
    # Apply lemmatization and stopword removal
    tokens = [lemmatizer.lemmatize(token) for token in raw_tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def extract_text_from_pdf(pdf_file):
    """
    Extract and clean text content from a PDF file.
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        str: Extracted and cleaned text from PDF, or None if extraction failed
    """
    try:
        with st.spinner("Processing PDF file..."):
            with io.BytesIO(pdf_file.read()) as pdf_stream:
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                text = ""
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += " " + page_text
                    except Exception as page_error:
                        st.warning(f"⚠️ Could not extract text from page {page_num + 1}")
                
                # Check if any text was extracted
                if not text.strip():
                    st.error("❌ No text could be extracted from the PDF.")
                    return None
                    
                # Clean and normalize text
                text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
                text = re.sub(r'[^\w\s@.,-]', '', text)  # Keep alphanumeric, spaces, and some special chars
                text = text.lower().strip()
                
                return text
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None
    finally:
        # Reset file pointer for potential reuse
        pdf_file.seek(0)

def compare_resume_with_job(resume_text, job_description):
    """
    Compare resume with job description using NLP techniques.
    
    Args:
        resume_text (str): Extracted text from resume
        job_description (str): Job description text
        
    Returns:
        dict: Analysis results including match percentage and keywords
    """
    # Validate inputs
    if not resume_text or not job_description:
        st.error("❌ Missing resume text or job description")
        return None

    try:
        with st.spinner("Analyzing resume and job description..."):
            # Process texts
            resume_tokens = process_text(resume_text)
            job_tokens = process_text(job_description)
            
            # Calculate term frequencies
            resume_freq = Counter(resume_tokens)
            job_freq = Counter(job_tokens)
            
            # Get all unique terms
            all_terms = set(resume_tokens + job_tokens)
            
            # Calculate cosine similarity
            dot_product = sum(resume_freq[term] * job_freq[term] for term in all_terms)
            resume_magnitude = math.sqrt(sum(freq * freq for freq in resume_freq.values()))
            job_magnitude = math.sqrt(sum(freq * freq for freq in job_freq.values()))
            
            if resume_magnitude and job_magnitude:
                similarity = dot_product / (resume_magnitude * job_magnitude)
            else:
                similarity = 0
            
            # Get top keywords
            resume_terms = resume_freq.most_common(10)
            job_terms = job_freq.most_common(10)
            
            # Validate results
            if not resume_terms or not job_terms:
                st.warning("⚠️ No significant keywords found in one or both documents")
            
            return {
                'match_percentage': similarity * 100,
                'resume_keywords': resume_terms,
                'job_keywords': job_terms
            }
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None

def create_keyword_visualization(results):
    """
    Create a keyword comparison visualization using Plotly.
    
    Args:
        results (dict): Analysis results with resume and job keywords
        
    Returns:
        plotly.graph_objects.Figure: Interactive visualization
    """
    # Get all unique keywords
    all_words = set()
    for word, _ in results['resume_keywords'][:8]:
        all_words.add(word)
    for word, _ in results['job_keywords'][:8]:
        all_words.add(word)
    all_words = list(all_words)
    
    # Create score dictionaries for easy lookup
    resume_dict = dict(results['resume_keywords'])
    job_dict = dict(results['job_keywords'])
    
    # Find maximum score for normalization
    max_score = max(
        max(score for _, score in results['resume_keywords'] or [(None, 0)]),
        max(score for _, score in results['job_keywords'] or [(None, 0)])
    )
    
    if max_score == 0:
        max_score = 1  # Prevent division by zero
    
    # Get scores for all words
    resume_scores = [resume_dict.get(word, 0) / max_score * 100 for word in all_words]
    job_scores = [job_dict.get(word, 0) / max_score * 100 for word in all_words]
    
    # Sort by combined importance
    combined_scores = [(word, r + j) for word, r, j in zip(all_words, resume_scores, job_scores)]
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top words
    top_words = [item[0] for item in combined_scores[:8]]
    final_resume_scores = [resume_dict.get(word, 0) / max_score * 100 for word in top_words]
    final_job_scores = [job_dict.get(word, 0) / max_score * 100 for word in top_words]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars with enhanced styling
    fig.add_trace(go.Bar(
        y=top_words,
        x=final_resume_scores,
        orientation='h',
        name='Resume Keywords',
        marker=dict(
            color='rgba(99, 102, 241, 0.8)',
            line=dict(color='rgba(99, 102, 241, 1)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=top_words,
        x=final_job_scores,
        orientation='h',
        name='Job Keywords',
        marker=dict(
            color='rgba(16, 185, 129, 0.8)',
            line=dict(color='rgba(16, 185, 129, 1)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>'
    ))
    
    # Enhanced layout
    fig.update_layout(
        title={
            'text': 'Keyword Relevance Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, family='Inter, sans-serif')
        },
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=20, r=20, t=100, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#1e293b'),
        barmode='group',
        bargap=0.2,
        bargroupgap=0.1
    )
    
    # Enhanced axes
    fig.update_xaxes(
        title=dict(
            text='Relevance Score (%)',
            font=dict(size=16, family='Inter, sans-serif')
        ),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.2)',
        zerolinewidth=1,
        range=[0, 100]
    )
    
    fig.update_yaxes(
        title=dict(
            text='Keywords',
            font=dict(size=16, family='Inter, sans-serif')
        ),
        showgrid=False,
        tickfont=dict(size=14)
    )
    
    return fig

def render_sidebar():
    """Render the sidebar with team information."""
    with st.sidebar:
        st.markdown("<div class='team-header'>NLP Project Team</div>", unsafe_allow_html=True)
        
        # Team members
        team_members = [
            {"name": "S.Bhanusri", "role": "Team Member"},
            {"name": "Ch.Manjusha", "role": "Team Member"},
            {"name": "K.Navitha", "role": "Team Member"},
        ]
        
        # Display team members
        for member in team_members:
            leader_class = " leader" if member["role"] == "Team Leader" else ""
            st.markdown(f"""
            <div class='team-member{leader_class}'>
                <div class='member-name'>{member["name"]}</div>
                <div class='member-role'>{member["role"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add instructions
        st.markdown("### How to use")
        st.markdown("""
        1. Paste the job description in the text area
        2. Upload your resume as a PDF file
        3. View your match score and keyword analysis
        """)

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Resume Matcher",
        page_icon="📄",
        layout="wide"
    )
    
    # Load CSS if available
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Define fallback styling
        st.markdown("""
        <style>
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1e293b;
            text-align: center;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #64748b;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 0.5rem;
        }
        .match-score {
            font-size: 2rem;
            font-weight: bold;
            color: #4f46e5;
            text-align: center;
            padding: 1rem;
            background: #f1f5f9;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 1.5rem 0;
        }
        .team-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #1e293b;
        }
        .team-member {
            padding: 0.75rem;
            background: #f1f5f9;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .team-member.leader {
            background: #ede9fe;
            border-left: 3px solid #8b5cf6;
        }
        .member-name {
            font-weight: 600;
            color: #334155;
        }
        .member-role {
            font-size: 0.9rem;
            color: #64748b;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Make sure NLTK data is available
    download_nltk_data()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.markdown("<h1 class='title'>Resume Matcher</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Resume Analysis</p>", unsafe_allow_html=True)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>Job Description</div>", unsafe_allow_html=True)
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Copy and paste the job description you're interested in applying for..."
        )
    
    with col2:
        st.markdown("<div class='section-header'>Your Resume</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF format only)",
            type=["pdf"],
            help="Please ensure your PDF is text-searchable for best results"
        )
    
    # Process when both inputs are provided
    if uploaded_file is not None and job_description:
        # Extract text from resume
        resume_text = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            # Compare resume with job description
            results = compare_resume_with_job(resume_text, job_description)
            
            if results:
                # Display match percentage
                match_score = results['match_percentage']
                st.markdown(
                    f"<div class='match-score'>Match Score: {match_score:.1f}%</div>",
                    unsafe_allow_html=True
                )
                
                # Display visualization
                fig = create_keyword_visualization(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional analysis
                st.markdown("### Analysis Summary")
                
                # Job Keywords
                st.markdown("#### Top Job Description Keywords")
                job_keywords = ", ".join([f"**{word}**" for word, _ in results['job_keywords'][:5]])
                st.markdown(f"The job is looking for: {job_keywords}")
                
                # Resume Keywords
                st.markdown("#### Top Resume Keywords")
                resume_keywords = ", ".join([f"**{word}**" for word, _ in results['resume_keywords'][:5]])
                st.markdown(f"Your resume highlights: {resume_keywords}")
                
                # Improvement suggestions
                if match_score < 70:
                    st.warning("#### Improvement Suggestions")
                    missing_keywords = set(word for word, _ in results['job_keywords'][:10]) - \
                                      set(word for word, _ in results['resume_keywords'][:10])
                    if missing_keywords:
                        st.markdown(f"Consider adding these keywords to your resume: **{', '.join(list(missing_keywords)[:5])}**")
                    st.markdown("Try to tailor your resume specifically to match the job description.")
                else:
                    st.success("Your resume is well-matched to this job description!")
                
    elif uploaded_file is not None or job_description:
        # Prompt for the missing input
        if not uploaded_file:
            st.info("Please upload your resume to see the analysis.")
        if not job_description:
            st.info("Please paste a job description to see the analysis.")
    else:
        # Initial instructions
        st.info("👆 Start by providing both a job description and your resume.")

if __name__ == "__main__":
    main()