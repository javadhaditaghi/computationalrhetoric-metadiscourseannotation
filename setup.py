# setup.py - Setup and verification script
import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def create_directory_structure():
    """Create required directory structure if it doesn't exist"""
    directories = [
        "result/claude",
        "result/gemini",
        "result/deepseek",
        "result/optimized",
        "src/annotation/prompt",
        "src/annotation/llm/optimizer",
        "util"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def check_env_file():
    """Check if .env file exists and has required variables"""
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating from template...")
        with open('.env', 'w') as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("✓ Created .env file. Please edit it and add your OpenAI API key.")
        return False

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key or api_key == 'your_openai_api_key_here':
        print("⚠️  Please set your OpenAI API key in the .env file")
        return False

    print("✓ OpenAI API key found in .env file")
    return True


def check_csv_files():
    """Check if CSV files exist in the required directories"""
    models = ['claude', 'gemini', 'deepseek']
    missing_files = []

    for model in models:
        csv_dir = f"result/{model}"
        if not os.path.exists(csv_dir):
            missing_files.append(f"{csv_dir}/ (directory)")
            continue

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            missing_files.append(f"{csv_dir}/*.csv")
        else:
            print(f"✓ Found {len(csv_files)} CSV file(s) in {csv_dir}/")

    if missing_files:
        print("⚠️  Missing CSV files:")
        for missing in missing_files:
            print(f"   - {missing}")
        return False

    return True


def check_prompt_files():
    """Check if prompt files exist"""
    prompt_files = [
        "src/annotation/prompt/optimization_prompt.txt",
        "src/annotation/prompt/internalexternal.txt"
    ]

    missing_prompts = []
    for prompt_file in prompt_files:
        if not os.path.exists(prompt_file):
            missing_prompts.append(prompt_file)
        else:
            print(f"✓ Found prompt file: {prompt_file}")

    if missing_prompts:
        print("⚠️  Missing prompt files:")
        for missing in missing_prompts:
            print(f"   - {missing}")

        # Create default optimization prompt if missing
        if "optimization_prompt.txt" in str(missing_prompts):
            create_default_optimization_prompt()

        return len(missing_prompts) == 0

    return True


def create_default_optimization_prompt():
    """Create a default optimization prompt file"""
    default_prompt = """You are an expert annotation optimizer. Your task is to analyze annotations from three different AI models (Claude, Gemini, and DeepSeek) and provide the most accurate final decision for metadiscourse classification.

Each annotation contains:
- role: Metadiscourse, Propositional, or Borderline  
- confidence: 1-5 scale (1=very uncertain, 5=very certain)
- note: Additional comments about secondary roles or uncertainty
- justification: Evidence-based rationale citing specific contextual or linguistic features
- context_assessment: Assessment of whether context is sufficient

Consider the following factors when making your final decision:
1. Confidence scores from each model (higher confidence should carry more weight)
2. Quality and specificity of justifications (detailed linguistic evidence is better)
3. Consistency across models (agreement suggests higher reliability)
4. Context assessment quality (better context understanding leads to better decisions)
5. Linguistic and contextual evidence provided in justifications

Metadiscourse expressions typically:
- Organize text structure (transitions, sequencing)
- Express attitude/evaluation (hedges, boosters, attitude markers)
- Engage with readers (reader references, questions)
- Clarify meaning (code glosses, reformulations)

Propositional expressions typically:
- Convey factual content
- Present research findings
- Describe methods or procedures
- State claims or arguments

Borderline cases may:
- Have both metadiscourse and propositional functions
- Be ambiguous due to insufficient context
- Require additional context for accurate classification

Provide your final decision in this exact JSON format:
{
    "role": "Metadiscourse|Propositional|Borderline",
    "confidence": 1-5,
    "note": "synthesis of key insights from all models and your reasoning",
    "justification": "comprehensive rationale based on all annotations and linguistic evidence",
    "context_assessment": "overall assessment of context sufficiency"
}"""

    os.makedirs("src/annotation/prompt", exist_ok=True)
    with open("src/annotation/prompt/optimization_prompt.txt", "w") as f:
        f.write(default_prompt)
    print("✓ Created default optimization prompt file")


def install_requirements():
    """Install required Python packages"""
    try:
        import pandas
        import openai
        import dotenv
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"⚠️  Missing required package: {e}")
        print("Please run: pip install pandas openai python-dotenv")
        return False


def run_quick_test():
    """Run a quick test to verify the setup works"""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            print("⚠️  Cannot run test - OpenAI API key not configured")
            return False

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Test API connection with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )

        print("✓ OpenAI API connection test successful")
        return True

    except Exception as e:
        print(f"⚠️  API test failed: {e}")
        return False


def check_file_structure():
    """Check if the code files are in the correct locations"""
    expected_files = [
        "src/annotation/llm/optimizer/gpt_annotator.py",
        "util/config.py",
        "run_optimizer.py"
    ]

    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ Found: {file_path}")

    if missing_files:
        print("⚠️  Missing code files:")
        for missing in missing_files:
            print(f"   - {missing}")
        return False

    return True


def main():
    """Main setup function"""
    print("🚀 Annotation Optimizer Setup")
    print("=" * 40)

    # Check and install requirements
    if not install_requirements():
        sys.exit(1)

    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()

    # Check file structure
    print("\n📄 Checking code file locations...")
    files_ok = check_file_structure()

    # Check .env file
    print("\n🔑 Checking environment configuration...")
    env_ok = check_env_file()

    # Check prompt files
    print("\n📝 Checking prompt files...")
    prompts_ok = check_prompt_files()

    # Check CSV files
    print("\n📊 Checking CSV data files...")
    csv_ok = check_csv_files()

    # Run API test if env is configured
    if env_ok:
        print("\n🔍 Testing OpenAI API connection...")
        api_ok = run_quick_test()
    else:
        api_ok = False

    # Summary
    print("\n" + "=" * 40)
    print("📋 SETUP SUMMARY")
    print("=" * 40)

    status_items = [
        ("Code file structure", "✓" if files_ok else "⚠️"),
        ("Environment file (.env)", "✓" if env_ok else "⚠️"),
        ("Prompt files", "✓" if prompts_ok else "⚠️"),
        ("CSV data files", "✓" if csv_ok else "⚠️"),
        ("OpenAI API connection", "✓" if api_ok else "⚠️")
    ]

    for item, status in status_items:
        print(f"{status} {item}")

    if all([files_ok, env_ok, prompts_ok, csv_ok, api_ok]):
        print(f"\n🎉 Setup complete! You can now run:")
        print(f"   python run_optimizer.py")
    else:
        print(f"\n⚠️  Setup incomplete. Please resolve the issues above before running the optimizer.")

        if not files_ok:
            print(f"\n💡 Make sure to place the code files in the correct locations:")
            print(f"   - gpt_annotator.py → src/annotation/llm/optimizer/")
            print(f"   - config.py → util/")

        if not env_ok:
            print(f"\n💡 Next steps:")
            print(f"   1. Edit the .env file and add your OpenAI API key")
            print(f"   2. Make sure you have CSV files in result/claude/, result/gemini/, and result/deepseek/")

        if not csv_ok:
            print(f"   3. Add your annotation CSV files to the appropriate directories")


if __name__ == "__main__":
    main()