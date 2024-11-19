class HebrewLetterValidator:
    """
    Validates the kosher status of individual Hebrew letters
    based on traditional scribal rules (Hilchot Stam)
    """
    
    # Dictionary of kosher letter formations
    KOSHER_LETTER_RULES = {
        'א': {
            'min_width': 0.3,
            'max_width': 0.7,
            'head_ratio': (0.4, 0.6),
            'body_straightness': 0.8
        },
        'ב': {
            'closed_top': True,
            'right_leg_angle': (30, 60),
            'left_leg_angle': (100, 140)
        },
        # Add more letters with their specific kosher rules
    }
    
    @classmethod
    def validate_letter(cls, letter_image, letter_char):
        """
        Validate a single letter's kosher status
        
        Args:
            letter_image (np.ndarray): Preprocessed letter image
            letter_char (str): Hebrew letter character
        
        Returns:
            dict: Validation results
        """
        if letter_char not in cls.KOSHER_LETTER_RULES:
            return {
                'is_kosher': False,
                'reason': 'No validation rules for this letter'
            }
        
        # Perform specific checks based on letter rules
        rules = cls.KOSHER_LETTER_RULES[letter_char]
        validation_results = {
            'letter': letter_char,
            'checks': []
        }
        
        # Example validation logic (to be expanded)
        # These are placeholder checks and would need sophisticated image analysis
        def check_width_ratio(image):
            # Placeholder width check
            width_ratio = image.shape[1] / image.shape[0]
            return (rules.get('min_width', 0) <= width_ratio <= 
                    rules.get('max_width', 1))
        
        validation_results['checks'].append({
            'name': 'Width Ratio',
            'passed': check_width_ratio(letter_image)
        })
        
        # Determine overall kosher status
        validation_results['is_kosher'] = all(
            check['passed'] for check in validation_results['checks']
        )
        
        return validation_results

    @classmethod
    def validate_scroll(cls, letters_with_chars):
        """
        Validate entire mezuzah scroll
        
        Args:
            letters_with_chars (list): List of tuples (letter_image, letter_char)
        
        Returns:
            dict: Overall scroll validation results
        """
        scroll_validation = {
            'total_letters': len(letters_with_chars),
            'kosher_letters': 0,
            'invalid_letters': [],
            'is_kosher': True
        }
        
        for letter_image, letter_char in letters_with_chars:
            letter_result = cls.validate_letter(letter_image, letter_char)
            
            if not letter_result['is_kosher']:
                scroll_validation['is_kosher'] = False
                scroll_validation['invalid_letters'].append({
                    'letter': letter_char,
                    'details': letter_result
                })
            else:
                scroll_validation['kosher_letters'] += 1
        
        scroll_validation['kosher_percentage'] = (
            scroll_validation['kosher_letters'] / 
            scroll_validation['total_letters']
        ) * 100
        
        return scroll_validation
