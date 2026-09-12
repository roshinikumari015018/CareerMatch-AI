import asyncio
from mcp import Client, StdioServerParameters


async def main():

    server = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    async with Client(server) as client:

        result = await client.call_tool(
            "get_job_details",
            {
                "job_title": "Java Developer"
            }
        )

        print("\n==============================")
        print("       JOB DETAILS")
        print("==============================\n")

        print(result)


if __name__ == "__main__":
    asyncio.run(main())