const { OpenAIEmbeddings } = require("@langchain/openai");
const { HttpsProxyAgent } = require("https-proxy-agent");

const axios = require("axios");
const { Configuration, OpenAIApi } = require("openai");
// https://js.langchain.com/v0.2/docs/integrations/text_embedding/openai/

const proxy = {
  host: "http://10.39.152.30", // e.g., 'proxy.example.com'
  port: 3128, // e.g., 8080
};
// Create an axios instance with proxy configuration
const axiosInstance = axios.create({
  proxy: proxy,
  // Optionally, you can add headers or other axios configurations here
});
// Configure OpenAI to use the axios instance
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
  baseOptions: {
    adapter: axiosInstance.defaults.adapter, // Set the axios instance
  },
});
const openai = new OpenAIApi(configuration);

/**
 *
 * @param {*} textChunks array of text chunks
 * @returns array of embeddings
 */
async function embedTexts(textChunks) {
  // const embedder = new OpenAIEmbeddings(
  //   {
  //     apiKey: process.env.OPENAI_API_KEY,
  //     batchSize: 512, // Default value if omitted is 512. Max is 2048
  //     model: "text-embedding-3-large",
  //   },
  //   {
  //     baseOptions: {
  //       proxy: false,
  //       httpAgent: new HttpsProxyAgent("http://10.39.152.30:3128"),
  //       httpsAgent: new HttpsProxyAgent("http://10.39.152.30:3128"),
  //     },
  //   }
  // );
  const embeddingsDataArr = []; //[{embedding: [], chunk: '}]

  for (const chunk of textChunks) {
    console.log("Embedding chunk", chunk);
    const embedding = await openai.createEmbedding({
      model: "text-embedding-3-large",
      input: chunk,
    });
    embeddingsDataArr.push({
      embedding,
      chunk,
    });
    console.log("Embedding value", embedding);
  }

  return embeddingsDataArr;
}

//const embeddings = await embedText(chunks);

module.exports = {
  embedTexts,
};
