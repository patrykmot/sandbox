package com.lovesoft.winner.web;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;

/** Single place for JSON (de)serialisation. Only the web layer uses JSON; the engine never does. */
public final class Json {

    /** Gson is thread-safe; nulls are written so the API always returns the same fields. */
    private static final Gson GSON = new GsonBuilder()
            .serializeNulls()
            .disableHtmlEscaping()
            .create();

    private Json() {
    }

    public static String toJson(Object value) {
        return GSON.toJson(value);
    }

    /**
     * Parses a request body. An empty body yields {@code null}.
     *
     * @throws BadRequestException if the body is not valid JSON for the given type
     */
    public static <T> T fromJson(String json, Class<T> type) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return GSON.fromJson(json, type);
        } catch (JsonParseException e) {
            throw new BadRequestException("Malformed JSON body");
        }
    }
}
