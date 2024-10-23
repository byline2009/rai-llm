const { OpenAIEmbeddings } = require("@langchain/openai");
const { HttpsProxyAgent } = require("https-proxy-agent");
// https://js.langchain.com/v0.2/docs/integrations/text_embedding/openai/

/**
 *
 * @param {*} textChunks array of text chunks
 * @returns array of embeddings
 */
async function embedTexts(textChunks) {
  const embedder = new OpenAIEmbeddings(
    {
      apiKey: process.env.OPENAI_API_KEY,
      batchSize: 512, // Default value if omitted is 512. Max is 2048
      model: "text-embedding-3-large",
    }
    // {
    //   baseOptions: {
    //     proxy: false,
    //     httpAgent: new HttpsProxyAgent("http://10.39.152.30:3128"),
    //     httpsAgent: new HttpsProxyAgent("http://10.39.152.30:3128"),
    //   },
    // }
  );
  const embeddingsDataArr = []; //[{embedding: [], chunk: '}]

  for (const chunk of textChunks) {
    console.log("Embedding chunk", chunk);
    const embedding = await embedder.embedQuery(chunk);
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
