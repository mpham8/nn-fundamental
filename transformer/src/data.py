import sqlite3
from pathlib import Path

#first copy over imessages from ~/Library/Messages/chat.db to data/chat.db
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB = _DATA_DIR / "chat.db"
DEFAULT_OUTPUT = _DATA_DIR / "self_imessages.txt"


def extract_self_imessages(
    output_file: Path | str = DEFAULT_OUTPUT,
    db_path: Path | str = DEFAULT_DB,
) -> int:
    """
    Extracts iMessages from solo-handle ("Notes to self") chats.
    Expects a copy of chat.db at data/chat.db (not ~/Library/Messages).
    """
    db_path = Path(db_path)
    output_file = Path(output_file)

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Copy your Messages DB here, e.g. cp ~/Library/Messages/chat.db data/chat.db"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT chat.ROWID, handle.id
        FROM chat
        JOIN chat_handle_join ON chat.ROWID = chat_handle_join.chat_id
        JOIN handle ON chat_handle_join.handle_id = handle.ROWID
        GROUP BY chat.ROWID
        HAVING COUNT(handle.id) = 1
        """
    )
    self_chats = cursor.fetchall()

    if not self_chats:
        print("No 'myself' messages found.")
        conn.close()
        return 0

    all_my_messages: list[str] = []
    for chat_rowid, _handle_id in self_chats:
        cursor.execute(
            """
            SELECT message.text
            FROM chat_message_join
            JOIN message ON chat_message_join.message_id = message.ROWID
            WHERE chat_message_join.chat_id = ?
            ORDER BY message.date
            """,
            (chat_rowid,),
        )
        for (msg_text,) in cursor.fetchall():
            if msg_text is None:
                continue
            all_my_messages.append(msg_text.strip())

    with open(output_file, "w", encoding="utf-8") as f:
        for msg in all_my_messages:
            if msg:
                f.write(msg.replace("\n", " ").strip() + "\n")

    conn.close()
    print(f"Extracted {len(all_my_messages)} 'myself' messages to {output_file}")
    return len(all_my_messages)


if __name__ == "__main__":
    extract_self_imessages()
