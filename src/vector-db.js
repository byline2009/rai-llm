const { Pinecone } = require("@pinecone-database/pinecone");

const { embedTexts } = require("./embed-texts");
const { Dispatcher, ProxyAgent } = require("undici");

const DB_INDEX = process.env.PINECONE_INDEX_NAME;
const NAMESPACE = process.env.PINECONE_NAME_SPACE;
const client = new ProxyAgent("http://10.39.152.30:3128");
const customFetch = (input, init) => {
  return fetch(input, {
    ...init,
    dispatcher: client,
    keepalive: true,
  });
};

const config = {
  apiKey: "409e625d-dec0-4241-88bc-30efca393b76",
  fetchApi: customFetch,
};
// https://docs.pinecone.io/guides/get-started/quickstart
const pc = new Pinecone(config);

/**
 *
 * @param {*} embeddings Array of embedding & chunk: [{embedding: [], chunk: ''}]
 * @param {*} namespace
 */
async function storeEmbeddings(embeddings, namespace = NAMESPACE) {
  const index = pc.index(DB_INDEX);

  for (let i = 0; i < embeddings.length; i++) {
    await index.namespace(namespace).upsert([
      {
        id: `chunk-${i}`,
        values: embeddings[i].embedding,
        metadata: { chunk: embeddings[i].chunk },
      },
    ]);
  }
}

const createIndex = async () => {
  await pc.createIndex({
    name: DB_INDEX,

    // should match embedding model name, e.g. 3072 for OpenAI text-embedding-3-large and 1536 for OpenAI text-embedding-ada-002
    dimension: 3072,
    metric: "cosine",
    spec: {
      serverless: {
        cloud: "aws",
        region: "us-east-1",
      },
    },
  });
  console.log("Index created", DB_INDEX);
};

async function checkIndexExists() {
  // List all indexes
  const response = await pc.listIndexes();
  const indexes = response.indexes;
  console.log("Available indexes:", indexes);

  // Check if the desired index is in the list
  return indexes.find((item) => item.name === DB_INDEX);
}

const describeIndexStats = async () => {
  const index = pc.index(DB_INDEX);
  const stats = await index.describeIndexStats();
  return stats;
};

// https://docs.pinecone.io/guides/data/query-data
async function retrieveRelevantChunks(query, namespace = NAMESPACE) {
  const embeddingDataArr = await embedTexts([query]);
  const index = pc.index(DB_INDEX);
  const results = await index.namespace(namespace).query({
    vector: embeddingDataArr[0].embedding,
    topK: 5, // Number of relevant chunks to retrieve
    includeValues: true,
    includeMetadata: true,
  });
  return results.matches.map((match) => match.metadata.chunk);
}

// Storing embeddings in Pinecone
//await storeEmbeddings(embeddings, 'your-namespace');

module.exports = {
  storeEmbeddings,
  createIndex,
  describeIndexStats,
  retrieveRelevantChunks,
  checkIndexExists,
};
