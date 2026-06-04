import { END,START, MessagesAnnotation, StateGraph } from '@langchain/langgraph';
import readline from 'node:readline/promises';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { ChatGroq } from '@langchain/groq';
import { TavilySearch } from '@langchain/tavily';
import dotenv from 'dotenv';
dotenv.config();


const tool = new TavilySearch({
    maxResults: 3,
    apiKey: process.env.TAVILY_API_KEY,
    topic: 'general'
})


const tools = [tool]
const toolNode = new ToolNode(tools)

// 1. define node function
// 2. build the graph
// 3. Compile and invoke the graph

//MODEL INITIALIZATION AS LLM
const llm = new ChatGroq({
    model:'llama-3.3-70b-versatile',
    temperature: 0.9,
    maxTries: 2
}).bindTools(tools)

async function callModel(state) {
    //call the llm using apis
    console.log("Calling the model")
    const response = await llm.invoke(state.messages)
    return {messages:[response]}
}

// Conditional Node function example
function shouldContinue(state) {
    console.log("the State", state)
    return END;
}

// BUILD THE GRAPH
const workflow = new StateGraph(MessagesAnnotation)
    .addNode("agent", callModel) 
    .addNode('tools',toolNode)
    .addEdge(START, 'agent')
    .addEdge('agent', END)
    .addEdge("tools","agent")

// COMPILE THE GRAPH

const app = workflow.compile()

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    while (true) {

        const userInput = await rl.question('You: ');
        console.log("You Said: ", userInput);
        if (userInput.toLowerCase() === 'exit') {
            console.log("Exiting...");
            break;
        }

        const finalState = await app.invoke({
            messages: [new HumanMessage(userInput)]
        });

        const lastMessage = finalState.messages[finalState.messages.length - 1]
        console.log("Final State: ", lastMessage.text);
    }
    rl.close();

}

main()