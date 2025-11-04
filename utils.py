import pdfplumber
import os
import logging
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import config

logger = logging.getLogger(__name__)


def extract_text_with_pypdf2(pdf_path):
    """Alternative extraction dengan PyPDF2"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logger.info(f"PyPDF2 extracted {len(text)} characters from {os.path.basename(pdf_path)}")
        return text if text.strip() else None
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF dengan method yang lebih baik"""
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Coba extract dengan different methods
                page_text = page.extract_text(
                    layout=False,  # False untuk teks simple, True untuk complex layout
                    x_tolerance=2,
                    y_tolerance=2
                )
                
                if not page_text:
                    # Fallback: coba dengan layout True
                    page_text = page.extract_text(layout=True)
                
                if page_text:
                    # Clean text lebih agresif
                    # Hapus karakter special kecuali huruf, angka, dan punctuation dasar
                    page_text = re.sub(r'[^\w\s.,!?;:()\-@/\u00A0-\u024F\u1E00-\u1EFF]', '', page_text)
                    # Normalize whitespace
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    # Hapus line breaks berlebihan
                    page_text = re.sub(r'\n+', ' ', page_text)
                    
                    if len(page_text) > 10:  # Minimal content threshold
                        # Tambah page information hanya jika ada konten substantial
                        full_text += f"Page {page_num}: {page_text}\n\n"
        
        logger.info(f"pdfplumber extracted {len(full_text)} characters from {os.path.basename(pdf_path)}")
        return full_text if full_text.strip() else None
        
    except Exception as e:
        logger.error(f"pdfplumber extraction failed for {pdf_path}: {str(e)}")
        return None


def extract_text_from_pdf_fallback(pdf_path):
    """Fallback PDF extraction method dengan approach berbeda"""
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Method 1: Extract text dengan preservasi layout minimal
                text = page.extract_text(
                    layout=True,
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False
                )
                
                if text and len(text.strip()) > 10:
                    # Cleanup basic
                    text = re.sub(r'\s+', ' ', text).strip()
                    full_text += f"{text}\n\n"
        
        return full_text if full_text.strip() else None
    except Exception as e:
        logger.error(f"pdfplumber fallback extraction failed for {pdf_path}: {str(e)}")
        return None


def process_pdfs(pdf_paths, collection_id):
    """Process PDFs dengan improved text splitting dan error handling"""
    documents = []
    successful_files = 0

    for pdf_path in pdf_paths:
        logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
        
        # Try multiple extraction methods secara berurutan
        text = None
        extraction_method = ""
        
        # Method 1: pdfplumber primary
        text = extract_text_from_pdf(pdf_path)
        extraction_method = "pdfplumber"
        
        # Method 2: pdfplumber fallback
        if not text:
            text = extract_text_from_pdf_fallback(pdf_path)
            extraction_method = "pdfplumber_fallback"
        
        # Method 3: PyPDF2 sebagai last resort
        if not text:
            text = extract_text_with_pypdf2(pdf_path)
            extraction_method = "pypdf2"
        
        if text:
            # Validasi kualitas teks yang di-extract
            if not validate_pdf_text_quality(text):
                logger.warning(f"Text quality poor from {pdf_path}, skipping")
                continue
                
            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "file_path": pdf_path,
                    "collection_id": collection_id,
                    "extraction_method": extraction_method
                }
            )
            documents.append(doc)
            successful_files += 1
            logger.info(f"Successfully processed {os.path.basename(pdf_path)} with {extraction_method} - {len(text)} characters")
        else:
            logger.error(f"All extraction methods failed for {pdf_path}")

    if not documents:
        logger.error("No text could be extracted from any PDF")
        return 0

    logger.info(f"Successfully extracted text from {successful_files}/{len(pdf_paths)} files")

    # Improved text splitting dengan parameters yang dioptimalkan
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n\n", 
            "\n", 
            ". ", 
            "! ", 
            "? ", 
            "; ", 
            ", ", 
            " ", 
            ""
        ],
        length_function=len,
        keep_separator=True,
        is_separator_regex=False
    )

    try:
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    except Exception as e:
        logger.error(f"Text splitting failed: {str(e)}")
        return 0

    if not chunks:
        logger.error("No chunks created from documents")
        return 0

    # Create vector store
    try:
        from processor import processor
        
        # Pastikan embeddings sudah initialized
        if not hasattr(processor, 'embeddings') or processor.embeddings is None:
            logger.error("Embeddings not initialized in processor")
            return 0

        vector_store = FAISS.from_documents(chunks, processor.embeddings)

        # Save vector store
        index_path = os.path.join(config.index_folder, collection_id)
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)

        logger.info(f"Created vector store with {len(chunks)} chunks for collection {collection_id}")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise


def validate_pdf_text_quality(text):
    """Validasi kualitas teks yang di-extract dari PDF"""
    if not text:
        return False
    
    # Cek panjang minimal
    if len(text.strip()) < 50:
        return False
    
    # Cek ratio karakter valid vs total
    valid_chars = re.findall(r'[\w\s.,!?;:()\-@]', text)
    valid_ratio = len(valid_chars) / len(text) if text else 0
    
    if valid_ratio < 0.7:  # Minimal 70% karakter valid
        return False
    
    # Cek apakah ada kata-kata meaningful (minimal 3 huruf)
    words = [word for word in text.split() if len(word) >= 3]
    if len(words) < 10:  # Minimal 10 kata meaningful
        return False
    
    # Cek corruption ratio (karakter aneh)
    corruption_pattern = r'[^\w\s.,!?;:()\-@/\u00A0-\u024F\u1E00-\u1EFF]'
    corrupt_chars = re.findall(corruption_pattern, text)
    corruption_ratio = len(corrupt_chars) / len(text) if text else 0
    
    if corruption_ratio > 0.1:  # Maksimal 10% karakter corrupt
        logger.warning(f"High corruption ratio: {corruption_ratio:.2f}")
        return False
    
    return True


def get_pdf_info(pdf_path):
    """Get basic info about PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            info = {
                "pages": len(pdf.pages),
                "file_size": os.path.getsize(pdf_path),
                "file_name": os.path.basename(pdf_path)
            }
            return info
    except Exception as e:
        logger.error(f"Failed to get PDF info for {pdf_path}: {str(e)}")
        return None


def test_pdf_extraction(pdf_path):
    """Test function untuk debug PDF extraction"""
    logger.info(f"Testing PDF extraction for: {pdf_path}")
    
    methods = {
        "pdfplumber_primary": extract_text_from_pdf,
        "pdfplumber_fallback": extract_text_from_pdf_fallback,
        "pypdf2": extract_text_with_pypdf2
    }
    
    results = {}
    for method_name, method_func in methods.items():
        try:
            text = method_func(pdf_path)
            if text:
                results[method_name] = {
                    "success": True,
                    "char_count": len(text),
                    "sample": text[:200] + "..." if len(text) > 200 else text,
                    "quality_score": validate_pdf_text_quality(text)
                }
            else:
                results[method_name] = {"success": False}
        except Exception as e:
            results[method_name] = {"success": False, "error": str(e)}
    
    return results