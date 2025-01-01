# backend/utils/corpus_manager.py
import json
import itertools
from pathlib import Path

class CorpusManager:
    def __init__(self, corpus_path="corpus_data.json"):
        self.corpus_path = Path(corpus_path)
        self.templates = {
            "analysis": "Let's examine {topic} from {perspective} perspectives",
            "strategy": "{action} drives {outcome} in {context}",
            "inquiry": "How might {subject} impact {domain} in {timeframe}",
            "evaluation": "What {aspect} should we consider when analyzing {focus}",
            "implementation": "To implement {solution}, we need to consider {factor} and {constraint}"
        }
        self.variables = {
            "topic": ["AI", "design", "communication", "technology", "innovation"],
            "perspective": ["technical", "strategic", "analytical", "practical", "theoretical"],
            "action": ["Innovation", "Collaboration", "Learning", "Transformation", "Integration"],
            "outcome": ["growth", "progress", "transformation", "efficiency", "optimization"],
            "context": ["modern environments", "digital landscapes", "global markets", "emerging ecosystems", "dynamic systems"],
            "subject": ["technological change", "market dynamics", "user behavior", "system architecture", "data patterns"],
            "domain": ["business processes", "user experience", "system performance", "team dynamics", "market position"],
            "timeframe": ["near future", "long term", "rapid deployment", "phased implementation", "continuous evolution"],
            "aspect": ["technical requirements", "user needs", "resource constraints", "performance metrics", "success factors"],
            "focus": ["system design", "user interaction", "data flow", "process optimization", "integration points"],
            "solution": ["new features", "system updates", "process changes", "architectural improvements", "optimization strategies"],
            "factor": ["technical debt", "user adoption", "resource allocation", "timeline constraints", "dependency management"],
            "constraint": ["system limitations", "budget restrictions", "time constraints", "technical capabilities", "team capacity"]
        }
        self.load_corpus()
    
    def load_corpus(self):
        with open(self.corpus_path) as f:
            self.corpus_data = json.load(f)
    
    def generate_from_template(self, template_name):
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        # Extract variable names without brackets
        template_vars = [var.split("}")[0] for var in template.split("{")[1:]]
        
        try:
            var_combinations = itertools.product(*[self.variables[var] for var in template_vars])
            return [template.format(**dict(zip(template_vars, combo))) 
                   for combo in var_combinations]
        except KeyError as e:
            raise ValueError(f"Variable {e} not found in variables dictionary")
    
    def get_all_sentences(self):
        sentences = []
        for category in self.corpus_data["categories"].values():
            if isinstance(category, dict):
                for subcategory in category.values():
                    if isinstance(subcategory, list):
                        sentences.extend(subcategory)
            elif isinstance(category, list):
                sentences.extend(category)
        return sentences
    
    def get_sentences(self, category=None, subcategory=None):
        if not category:
            return list(self.get_all_sentences())
        if category not in self.corpus_data["categories"]:
            raise ValueError(f"Category '{category}' not found")
        if not subcategory:
            return self.corpus_data["categories"][category]
        if subcategory not in self.corpus_data["categories"][category]:
            raise ValueError(f"Subcategory '{subcategory}' not found in category '{category}'")
        return self.corpus_data["categories"][category][subcategory]
