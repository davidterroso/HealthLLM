import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_functions import embed_docs

def fetch_pmc_articles_by_query(query: str, page_size: int=25):
        """
        Given a query, searches the open-access
        articles in the PMC website. This function
        is more oriented for testing, since it only
        allows the retrival of data using a query

        Args:
                query (str): search query in the PMC
                website
                page_size (int): number of articles
                per page

        Returns:
                results (List[dict{str}]):
                results of the search query.
                Returns a dictionary that stores
                the title, abstract, link, and
                PMID
        """

        # URL where we can access the articles published in PMC
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        # Parameters regarding which text will be retrieved
        params = {
                "query": f"{query} AND OPEN_ACCESS:Y",
                "format": "json",
                "pageSize": page_size,
                "resultType": "core"
        }

        response = requests.get(url=url, params=params)
        data = response.json()
        articles = data.get("resultList", {}).get("result", [])

        results = []

        # Iterate through the articles in the page
        for article in articles:
                entry = {
                "title": article.get("title"),
                "abstract": article.get("abstractText"),
                "source": article.get("fullTextUrlList", {}).get("fullTextUrl", []),
                "pmid": article.get("id")
                }

                results.append(entry)

        text_chunker(results=results)

def text_chunker(results: dict):
        """
        Function used to chunk the text
        present in the results that it
        is fed

        Args:
                results (List[dict{str}]):
                results of the search query.
                Represented in a dictionary
                that stores the title,
                abstract, link, and PMID
        """

        # Chunks the text so that it can be ingested into a vector database
        chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = chunker.create_documents([result["abstract"] for result in results if result.get("abstract")])
        embed_docs(docs)

# Being used to test function so far
query="diabetes"
fetch_pmc_articles_by_query(query="diabetes", page_size=25)
