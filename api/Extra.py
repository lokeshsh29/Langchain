

prompt1= ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2= ChatPromptTemplate.from_template("Write me a poem on {topic} with 100 words")

chain1 = prompt1 | GeminiModel
chain2 = prompt2 | llm

add_routes(
    app,
    chain1,
    path='/essay'    
)

add_routes(
    app,
    chain2,
    path='/poem'    
)