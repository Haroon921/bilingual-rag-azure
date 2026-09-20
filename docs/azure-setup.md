# Azure setup

> Model names, versions, SKUs and regional availability change often. Treat these commands as a
> starting point and confirm current options in the Azure portal or with `az cognitiveservices model list`.

You need one **Azure OpenAI** resource (a chat deployment and an embedding deployment) and one
**Azure AI Search** service. The Basic tier is enough; the demo corpus is tiny.

```bash
az login
RG=rg-bilingual-rag
LOC=swedencentral            # pick a region where your chosen models are available
AOAI=<unique-openai-name>
SEARCH=<unique-search-name>

az group create -n $RG -l $LOC

# Azure OpenAI resource (custom subdomain is required for Entra ID auth)
az cognitiveservices account create -n $AOAI -g $RG -l $LOC \
  --kind OpenAI --sku S0 --custom-domain $AOAI

# Model deployments: use the deployment NAMES you put in .env
# (check the portal for currently available model names/versions/SKUs)
az cognitiveservices account deployment create -g $RG -n $AOAI \
  --deployment-name gpt-4o --model-name gpt-4o --model-version "<version>" \
  --model-format OpenAI --sku-name GlobalStandard --sku-capacity 50
az cognitiveservices account deployment create -g $RG -n $AOAI \
  --deployment-name text-embedding-3-large --model-name text-embedding-3-large --model-version "1" \
  --model-format OpenAI --sku-name Standard --sku-capacity 50

# Azure AI Search with Entra ID (RBAC) enabled alongside keys
az search service create -n $SEARCH -g $RG -l $LOC --sku basic \
  --auth-options aadOrApiKey --aad-auth-failure-mode http401WithBearerChallenge
```

## Keyless access (recommended)

Grant your own identity the data-plane roles, then leave both API key variables empty in `.env`:

```bash
ME=$(az ad signed-in-user show --query id -o tsv)
SUB=$(az account show --query id -o tsv)
SCOPE=/subscriptions/$SUB/resourceGroups/$RG

az role assignment create --assignee $ME --role "Cognitive Services OpenAI User" \
  --scope $SCOPE/providers/Microsoft.CognitiveServices/accounts/$AOAI
az role assignment create --assignee $ME --role "Search Service Contributor" \
  --scope $SCOPE/providers/Microsoft.Search/searchServices/$SEARCH      # create the index
az role assignment create --assignee $ME --role "Search Index Data Contributor" \
  --scope $SCOPE/providers/Microsoft.Search/searchServices/$SEARCH      # upload documents
az role assignment create --assignee $ME --role "Search Index Data Reader" \
  --scope $SCOPE/providers/Microsoft.Search/searchServices/$SEARCH      # query
```

Role assignments can take a few minutes to propagate. When deploying the API to Azure, give its
**managed identity** the same roles instead of using keys.

## Clean up

```bash
az group delete -n $RG --yes --no-wait
```
