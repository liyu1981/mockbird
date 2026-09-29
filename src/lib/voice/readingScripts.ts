/**
 * Reading scripts for voice-cloning samples.
 *
 * Why a fixed script instead of "say anything": a consistent paragraph gives the
 * same phonemes and prosody every time, so the audio codes capture the speaker
 * rather than the topic, and it makes A/B comparisons between two recordings
 * meaningful. The model supports 20 languages, so there is one script per
 * language — read the one in the language your voice actually speaks.
 *
 * Each script is ~15–20 s at a natural pace, which is the sweet spot: long
 * enough to capture intonation, short enough that the prompt stays compact.
 */

export type ReadingScript = {
  /** ISO 639-1 code, matching the languages MOSS-TTS-Nano supports. */
  code: string;
  /** Name in English, for the picker. */
  label: string;
  /** Endonym, so speakers can find their own language. */
  native: string;
  /** Text direction; Arabic and Persian are right-to-left. */
  dir: "ltr" | "rtl";
  text: string;
};

export const READING_SCRIPTS: ReadingScript[] = [
  {
    code: "en",
    label: "English",
    native: "English",
    dir: "ltr",
    text: "Hello, and welcome to this voice sample. I am reading a short paragraph at a natural speed, with a small pause after every sentence. Please keep the room quiet and avoid background noise. That is all for today. Thank you for listening.",
  },
  {
    code: "zh",
    label: "Chinese",
    native: "中文",
    dir: "ltr",
    text: "你好，欢迎使用声音克隆功能。我正在用自然的速度朗读一小段文字，每句话之间会有短暂的停顿。请保持环境安静，避免背景噪音。这段录音大约需要十秒钟。非常感谢你的配合。",
  },
  {
    code: "ja",
    label: "Japanese",
    native: "日本語",
    dir: "ltr",
    text: "こんにちは。これは音声クローンのサンプルです。普通の速さで、少し長めの文章を読み上げます。文と文のあいだには、小さな間を入れます。背景のノイズや音楽は入れないでください。今日はここまでです。ありがとうございました。",
  },
  {
    code: "ko",
    label: "Korean",
    native: "한국어",
    dir: "ltr",
    text: "안녕하세요. 음성 복제 샘플입니다. 자연스러운 속도로 문장을 읽겠습니다. 문장과 문장 사이에는 잠깐 멈춤이 있습니다. 주변 소음은 넣지 마세요. 여기까지입니다. 감사합니다.",
  },
  {
    code: "de",
    label: "German",
    native: "Deutsch",
    dir: "ltr",
    text: "Guten Tag. Dies ist eine kurze Stimmprobe für die Sprachklonierung. Ich lese einen Absatz in natürlichem Tempo vor und mache zwischen den Sätzen eine kleine Pause. Bitte sorgen Sie für eine ruhige Umgebung ohne Hintergrundgeräusche. Das war alles, vielen Dank.",
  },
  {
    code: "fr",
    label: "French",
    native: "Français",
    dir: "ltr",
    text: "Bonjour. Voici un court échantillon de voix pour le clonage vocal. Je lis un paragraphe à vitesse naturelle, en faisant une petite pause entre les phrases. Essayez de rester dans un environnement calme, sans bruit de fond. C'est tout, merci beaucoup.",
  },
  {
    code: "es",
    label: "Spanish",
    native: "Español",
    dir: "ltr",
    text: "Hola. Esta es una muestra de voz corta para la clonación de voz. Leo un párrafo a velocidad natural, con una pequeña pausa entre cada frase. Intenta estar en un ambiente tranquilo, sin ruido de fondo. Eso es todo por hoy. Muchas gracias.",
  },
  {
    code: "pt",
    label: "Portuguese",
    native: "Português",
    dir: "ltr",
    text: "Olá. Esta é uma amostra de voz curta para clonagem de voz. Vou ler um parágrafo em velocidade natural, com uma pequena pausa entre as frases. Tente ficar em um ambiente silencioso, sem ruído de fundo. É isso por hoje. Muito obrigado.",
  },
  {
    code: "ru",
    label: "Russian",
    native: "Русский",
    dir: "ltr",
    text: "Здравствуйте. Это короткий образец голоса для клонирования. Я читаю абзац в естественном темпе, делая небольшую паузу между предложениями. Пожалуйста, записывайте в тихой комнате, без фонового шума. На этом всё, большое спасибо.",
  },
  {
    code: "it",
    label: "Italian",
    native: "Italiano",
    dir: "ltr",
    text: "Buongiorno. Questo è un breve campione di voce per la clonazione vocale. Leggo un paragrafo a velocità naturale, con una piccola pausa tra le frasi. Prova a restare in un ambiente tranquillo, senza rumori di fondo. È tutto per oggi. Grazie mille.",
  },
  {
    code: "pl",
    label: "Polish",
    native: "Polski",
    dir: "ltr",
    text: "Dzień dobry. To jest krótka próbka głosu do klonowania głosu. Czytam akapit w naturalnym tempie, robiąc krótką przerwę między zdaniami. Postaraj się, aby w tle nie było żadnych szumów. To wszystko na dziś. Dziękuję.",
  },
  {
    code: "cs",
    label: "Czech",
    native: "Čeština",
    dir: "ltr",
    text: "Dobrý den. Toto je krátká hlasová ukázka pro klonování hlasu. Čtu odstavec přirozenou rychlostí s krátkou pauzou mezi větami. Zkuste být v tichém prostředí bez šumů v pozadí. To je všechno. Děkuji vám.",
  },
  {
    code: "sv",
    label: "Swedish",
    native: "Svenska",
    dir: "ltr",
    text: "Hej. Det här är en kort röstprov för röstkloning. Jag läser ett stycke i naturligt tempo med en liten paus mellan meningarna. Försök att vara i en lugn miljö utan bakgrundsljud. Det var allt för idag. Tack så mycket.",
  },
  {
    code: "da",
    label: "Danish",
    native: "Dansk",
    dir: "ltr",
    text: "Hej. Dette er et kort stemmeprøve til stemmekloning. Jeg læser et afsnit i naturligt tempo med en lille pause mellem sætningerne. Forsøg at være i et roligt miljø uden baggrundsstøj. Det var alt for i dag. Mange tak.",
  },
  {
    code: "el",
    label: "Greek",
    native: "Ελληνικά",
    dir: "ltr",
    text: "Γεια σας. Αυτό είναι ένα σύντομο δείγμα φωνής για κλωνοποίηση. Διαβάζω μια παράγραφο με φυσικό ρυθμό, με μια μικρή παύση ανάμεσα στις προτάσεις. Προσπαθήστε να είστε σε ήσυχο περιβάλλον, χωρίς θόρυβο στο παρασκήνιο. Ήταν όλα. Ευχαριστώ πολύ.",
  },
  {
    code: "tr",
    label: "Turkish",
    native: "Türkçe",
    dir: "ltr",
    text: "Merhaba. Bu, ses klonlama için kısa bir ses örneğidir. Doğal bir hızda, cümleler arasında küçük bir duraklama yaparak bir paragraf okuyorum. Lütfen arka plan gürültüsü olmayan sakin bir ortamda kayıt yapın. Bugünlük bu kadar. Çok teşekkürler.",
  },
  {
    code: "hu",
    label: "Hungarian",
    native: "Magyar",
    dir: "ltr",
    text: "Szia. Ez egy rövid hangminta a hangklónozáshoz. Természetes sebességgel olvasok egy bekezdést, a mondatok között rövid szünettel. Kérlek, háttérzaj nélküli csendes helyen rögzíts. Ennyi volt. Köszönöm szépen.",
  },
  {
    code: "ar",
    label: "Arabic",
    native: "العربية",
    dir: "rtl",
    text: "مرحبًا. هذه عينة صوتية قصيرة لاستنساخ الصوت. سأقرأ فقرة بسرعة طبيعية، مع وقفة قصيرة بين كل جملتين. من فضلك سجّل في بيئة هادئة دون ضوضاء في الخلفية. هذا كل شيء. شكرًا جزيلًا.",
  },
  {
    code: "fa",
    label: "Persian",
    native: "فارسی",
    dir: "rtl",
    text: "سلام. این یک نمونهٔ صوتی کوتاه برای شبیه‌سازی صدا است. یک پاراگراف را با سرعت طبیعی می‌خوانم و میان جمله‌ها مکث کوتاهی دارم. لطفاً در محیطی آرام و بدون نویز پس‌زمینه ضبط کنید. همین بود. سپاسگزارم.",
  },
];

export const DEFAULT_SCRIPT_CODE = "en";

export function getReadingScript(code: string): ReadingScript {
  return READING_SCRIPTS.find((script) => script.code === code) ?? READING_SCRIPTS[0];
}

/**
 * Speaking rate, in syllables per second, for scripts that do not separate words
 * with spaces. Korean is spoken noticeably faster than Mandarin or Japanese, so
 * it gets its own figure rather than being lumped in with them.
 */
const CJK_SPEECH_RATE: Record<string, number> = { zh: 4.2, ja: 5.5, ko: 6.2 };

/**
 * Rough speaking time for a script, used only as a guidance hint. Latin-script
 * languages run at ~150 words per minute; CJK counts characters instead of words.
 */
export function estimateReadingSeconds(script: Pick<ReadingScript, "code" | "text">): number {
  const { code, text } = script;
  const cjk = (
    text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) ??
    []
  ).length;
  const words = text.split(/\s+/).filter(Boolean).length;
  const cjkSeconds = cjk / (CJK_SPEECH_RATE[code] ?? 5);
  const wordSeconds = words * 0.4;
  return Math.round(Math.max(4, (cjkSeconds + wordSeconds) * 10) / 10);
}
