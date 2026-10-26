from flake8.style_guide import DecisionEngine
from flake8.options.manager import OptionManager
from flake8.style_guide import Selected, Ignored

# Create options with explicit selections
options = OptionManager().get_default_values()
options.select = ['E111', 'E121']
options.ignore = []
options.extend_ignore = None

decider = DecisionEngine(options)

print(f"E111 decision: {decider.decision_for('E111')}")  # Expected: Selected.Explicitly
print(f"E112 decision: {decider.decision_for('E112')}")  # Expected: Ignored.Implicitly
