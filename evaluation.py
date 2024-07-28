import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision, context_recall, context_entity_recall, answer_relevancy


os.environ["OPENAI_API_KEY"] = "XXXXXXXXX"


data_samples = {
    'question': [
        'What year did World War II end?', 
        'Which country has the largest population?',
        'Who wrote "To Kill a Mockingbird"?',
        'What is the largest planet in our solar system?',
        'Who painted the Mona Lisa?'
    ],
    'answer': [
        'World War II ended in 1945', 
        'China has the largest population in the world',
        'Harper Lee wrote "To Kill a Mockingbird"',
        'Jupiter is the largest planet in our solar system',
        'Leonardo da Vinci painted the Mona Lisa'
    ],
    'contexts': [
        [
            'World War II, which lasted from 1939 to 1945, was a global conflict that involved most of the world\'s nations.',
            
        ], 
        [
            'Chinahas 1.4 billion people.',
            'India is the second most populous country with over 1.3 billion people.',
            'China has exceeding 1.4 billion people, is the most populous country in the world.'
        ],
        [
            '"To Kill a Mockingbird" is a novel by Harper Lee published in 1960.',
            'It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.',
            'Harper Lee, whose full name is Nelle Harper Lee, was an American novelist known for this work.'
        ],
        [
            'Jupiter is the largest planet in our solar system, known for its Great Red Spot and many moons.',
            'It is a gas giant with a mass one-thousandth that of the sun.',
            'Jupiter has a diameter of about 142,984 kilometers, making it the largest planet by far in our solar system.'
        ],
        [
            'The Mona Lisa is a portrait painting by the Italian artist Leonardo da Vinci.',
            'It is considered one of the most famous paintings in the world.',
            'Leonardo da Vinci was an Italian polymath of the Renaissance whose areas of interest included invention, painting, sculpting, architecture, science, music, and more.'
        ]
    ],
    'ground_truth': [
        'World War II ended in 1945', 
        'China has the largest population in the world',
        'Harper Lee wrote "To Kill a Mockingbird"',
        'Jupiter is the largest planet in our solar system',
        'Leonardo da Vinci painted the Mona Lisa'
    ]
}

dataset = Dataset.from_dict(data_samples)


metrics = [
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_relevancy,
]


score = evaluate(dataset, metrics=metrics)
df = score.to_pandas()
df.to_csv('d:/1_Full_Time_Job/prompt/Chatbot/score.csv', index=False)
