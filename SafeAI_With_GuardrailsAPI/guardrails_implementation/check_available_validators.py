# check_available_validators.py
import guardrails
from guardrails import validators
import pkgutil

import guardrails.version

print("Guardrails version:", guardrails.version.GUARDRAILS_VERSION)
print("\nAvailable validators:")

# Check what's available in the validators module
try:
    for importer, modname, ispkg in pkgutil.iter_modules(validators.__path__):
        print(f"- {modname}")
except:
    pass

# Check direct attributes
print("\nDirect validator attributes:")
for attr in dir(validators):
    if not attr.startswith('_'):
        print(f"- {attr}")

# Try to import common validators
common_validators = [
    'ProfanityFree', 'ToxicLanguage', 'SafeText', 'ValidLength', 
    'RegexMatch', 'OneLine', 'BugFree', 'ReadingTime'
]

print("\nTesting common validator imports:")
for validator in common_validators:
    try:
        exec(f"from guardrails.validators import {validator}")
        print(f"✓ {validator} - Available")
    except ImportError:
        print(f"✗ {validator} - Not available")

# Check if we can import Guard and basic functionality
try:
    from guardrails import Guard
    print("\n✓ Guard class is available")
    
    # Check Guard methods
    guard_methods = [method for method in dir(Guard) if not method.startswith('_')]
    print("Available Guard methods:", guard_methods[:10], "...")  # Show first 10
    
except ImportError as e:
    print(f"\n✗ Error importing Guard: {e}")