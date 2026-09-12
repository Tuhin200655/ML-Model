from predict import predict_condition, predict_from_voice

def main():
    print("==========================================")
    print("   Condition Detection Model Tester       ")
    print("==========================================\n")
    print("This tool allows you to test the ML model using text or voice.")
    print("Type 'exit' or 'quit' at any time to stop.\n")

    while True:
        print("\nChoose an input method:")
        print("1. Text Input")
        print("2. Voice Input (Microphone)")
        print("3. Exit")

        choice = input("\nSelect an option (1-3): ").strip()

        if choice.lower() in ['3', 'exit', 'quit']:
            print("Exiting tester. Goodbye!")
            break

        elif choice == '1':
            user_text = input("Enter the text you want to analyze: ").strip()
            if not user_text:
                print("Error: Text cannot be empty.")
                continue

            print("\nAnalyzing...")
            result = predict_condition(user_text)

            if isinstance(result, dict):
                print("\n--- Result ---")
                print(f"Label:       {result['label']}")
                print(f"Probability: {result['probability']:.4f}")
                print(f"Prediction:  {result['prediction']}")
                print(f"Threshold:   {result['threshold_used']}")
                print("--------------\n")
            else:
                print(f"Error: {result}")

        elif choice == '2':
            print("\nPreparing microphone... please wait.")
            # This calls the function in predict.py that uses SpeechRecognition
            result = predict_from_voice()

            if isinstance(result, dict):
                print("\n--- Result ---")
                print(f"Label:       {result['label']}")
                print(f"Probability: {result['probability']:.4f}")
                print(f"Prediction:  {result['prediction']}")
                print(f"Threshold:   {result['threshold_used']}")
                print("--------------\n")
            else:
                print(f"Error: {result}")

        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
