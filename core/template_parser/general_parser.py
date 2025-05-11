class GeneralParser:
    """
    A general parser that follows the same architecture as other parsers
    but simply returns the input text without specialized parsing.
    """
    
    def __init__(self):
        """Initialize the general parser."""
        self.parser_name = "general_parser"
    
    def parse(self, ocr_result, **kwargs):
        """
        Parse the input text without any specific extraction.
        
        Args:
            text (str): The input text to parse
            **kwargs: Additional keyword arguments
            
        Returns:
            dict: The parsed results containing the original text
        """
        # Perform basic preprocessing
        # processed_text = self.preprocess(text)
        
        # For general parser, we simply return the text as-is
        result = {
            "raw_text": ocr_result,
            "processed_text": ocr_result,
            "parser_type": self.parser_name,
            "extracted_data": {}
        }
        
        return self.postprocess(result)
    
    def preprocess(self, text):
        """
        Perform basic preprocessing on the input text.
        
        Args:
            text (str): The text to preprocess
            
        Returns:
            str: The preprocessed text
        """
        if text is None:
            return ""
        return text.strip()
    
    def postprocess(self, result):
        """
        Perform any postprocessing on the parsing results.
        
        Args:
            result (dict): The parsing results
            
        Returns:
            dict: The postprocessed results
        """
        return result