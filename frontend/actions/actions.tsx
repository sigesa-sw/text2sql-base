"use server";
import axios from "axios";

export async function generateQuestions() {
  const response = await axios.get(
    `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v0/generate_questions`
  );
  return response.data;
}

export async function generateSQL(question: string) {
  const response = await axios.get(
    `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v0/generate_sql`,
    {
      params: { question },
    }
  );
  return response.data;
}

export async function runSQL(cacheId: string, sql: string) {
  const response = await axios.get(
    `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v0/run_sql`,
    {
      params: { cacheId, sql },
    }
  );

  console.log("run", response.data);

  return response.data;
}