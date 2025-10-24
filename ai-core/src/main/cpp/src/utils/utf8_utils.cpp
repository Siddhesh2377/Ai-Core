#include "utf8_utils.h"
#include "logger.h"

#include <cassert>

namespace utf8 {

/* --------------------------------------------------------------------
 *  Thread‑local buffer that keeps incomplete UTF‑8 bytes while streaming
 * -------------------------------------------------------------------- */
    static thread_local std::string t_carry;

    inline std::string& get_carry_buffer() { return t_carry; }
    inline void clear_carry_buffer() { t_carry.clear(); }

/* --------------------------------------------------------------------
 *  Convert a Java string (UTF‑16) to a UTF‑8 std::string.
 * -------------------------------------------------------------------- */
    std::string from_jstring(JNIEnv* env, jstring js) {
        if (!js) return {};

        jsize len = env->GetStringLength(js);
        const jchar* chars = env->GetStringChars(js, nullptr);

        /*  `jchar` is `unsigned short`.  `std::u16string` expects
            `const char16_t*`.  An explicit cast keeps clang happy.    */
        std::u16string u16(reinterpret_cast<const char16_t*>(chars),
                           static_cast<size_t>(len));

        env->ReleaseStringChars(js, chars);

        /* very lightweight UTF‑16 → UTF‑8 conversion –  for the
           current use‑case a simple reinterpret cast is sufficient.
           If you need full Unicode correctness replace this block. */
        std::string out;
        out.reserve(u16.size() * 2);                 // rough upper‑bound

        for (char16_t cp : u16) {
            if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
            else if (cp <= 0x7ff) {
                out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
            } else {
                out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
            }
        }
        return out;
    }

/* --------------------------------------------------------------------
 *  Convert a UTF‑8 string to a Java string (UTF‑16).
 * -------------------------------------------------------------------- */
    jstring to_jstring(JNIEnv* env,
                       const std::string& utf8,
                       std::string& /*carry_buffer*/)
    {
        /* naive UTF‑8 → UTF‑16 –  our embedding / turnaround strings
           are all ASCII in practice, so this is fine. */
        auto len = static_cast<jsize>(utf8.size());
        auto* buf = new jchar[len];
        for (jsize i = 0; i < len; ++i) buf[i] = static_cast<jchar>(utf8[i]);

        jstring r = env->NewString(buf, len);
        delete[] buf;
        return r;
    }

/* --------------------------------------------------------------------
 *  Flush any remaining UTF‑8 bytes from the carry buffer as a
 *  replacement‑char (U+FFFD).
 * -------------------------------------------------------------------- */
    void flush_carry(JNIEnv* env, jobject cb) {
        if (t_carry.empty()) return;

        std::string tmp = t_carry + "\xEF\xBF\xBD"; // U+FFFD
        t_carry.clear();

        jclass cls = env->GetObjectClass(cb);
        if (!cls) return;
        jmethodID mid = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
        if (!mid) return;

        jstring js = to_jstring(env, tmp, t_carry);
        env->CallVoidMethod(cb, mid, js);
        env->DeleteLocalRef(js);
    }

} // namespace utf8