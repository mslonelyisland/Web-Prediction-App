
from django import forms  
from .models import MilkNew  

#forms.py
class MilkNewForm(forms.ModelForm):  
    class Meta:  
        model = MilkNew  
        fields = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade'] 
        widgets = { 
            'pH' : forms.NumberInput(attrs={'class': 'form-control'}),
            'Temperature' : forms.NumberInput(attrs={'class': 'form-control'}),
            'Taste' : forms.NumberInput(attrs={'class': 'form-control'}), 
            'Odor' : forms.NumberInput(attrs={'class': 'form-control'}), 
            'Fat' : forms.NumberInput(attrs={'class': 'form-control'}),
            'Turbidity' : forms.NumberInput(attrs={'class': 'form-control'}),
            'Colour' : forms.NumberInput(attrs={'class': 'form-control'})

      }
