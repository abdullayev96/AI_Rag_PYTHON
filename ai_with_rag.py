import chromadb
from sentence_transformers import SentenceTransformer

#  Embedding modelini yuklab olamiz
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#  Vektor bazasini ishga tushiramiz (Xotirada ishlaydigan baza)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs")

#  Kompaniyangizning bazadagi hujjatlari (Hujjatlarni chunklarga bo'lib kiritamiz)
documents = [
    "Kompaniyamizda ish vaqti soat 09:00 dan 18:00 gacha.",
    "Xodimlar yiliga 21 ish kuni to'lanadigan ta'til olish huquqiga ega.",
    "Masofaviy ishlash (Remote) haftasiga 2 kun ruxsat etiladi.",
]

# Hujjatlarni embedding qilish va baza qo'shish
collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3"],
)

#  Foydalanuvchi savoli
user_question = "Ta'til kunlari nechta?"

#  Savol uchun embedding olib, baza ichidan o'xshashini qidiramiz (Similarity Search)
results = collection.query(query_texts=[user_question], n_results=1)

relevant_chunk = results["documents"][0][0]
print(f"Topilgan eng mos hujjat: {relevant_chunk}")

#  LLM ga kontekst sifatida uzatish (Namuna)
prompt = f"""
Quyidagi ma'lumotga asoslanib savolga javob ber:
Ma'lumot: {relevant_chunk}
Savol: {user_question}
"""
print("\nTayyorlangan Prompt LLM uchun:\n", prompt)