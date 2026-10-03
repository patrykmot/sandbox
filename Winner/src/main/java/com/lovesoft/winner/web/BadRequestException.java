package com.lovesoft.winner.web;

/** Malformed HTTP input (bad JSON, bad id, body too large). Mapped to HTTP 400. */
public class BadRequestException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public BadRequestException(String message) {
        super(message);
    }
}
