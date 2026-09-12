import asyncio
from mcp import Client, StdioServerParameters


async def call_mcp_tool(tool_name, arguments):

    server = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with Client(server) as client:

        result = await client.call_tool(
            tool_name,
            arguments
        )

        return result


def search_jobs(query):
    return asyncio.run(
        call_mcp_tool(
            "search_jobs",
            {"query": query}
        )
    )


def get_job_details(job_title):
    return asyncio.run(
        call_mcp_tool(
            "get_job_details",
            {"job_title": job_title}
        )
    )


def analyze_skill_gap(job_title, user_skills):
    return asyncio.run(
        call_mcp_tool(
            "analyze_skill_gap",
            {
                "job_title": job_title,
                "user_skills": user_skills
            }
        )
    )


def get_learning_path(job_title):
    return asyncio.run(
        call_mcp_tool(
            "get_learning_path",
            {"job_title": job_title}
        )
    )
def recommend_jobs(user_skills):
    return asyncio.run(
        call_mcp_tool(
            "recommend_jobs",
            {"user_skills": user_skills}
        )
    )