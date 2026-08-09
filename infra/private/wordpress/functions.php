<?php
// infra/private/wordpress/functions.php

function wsmart_route_wp_setup() {
    // Add default posts and comments RSS feed links to head.
    add_theme_support( 'automatic-feed-links' );

    // Enable support for Post Thumbnails on posts and pages.
    add_theme_support( 'post-thumbnails' );

    // Register primary navigation menu.
    register_nav_menus(
        array(
            'menu-1' => esc_html__( 'Primary', 'wsmart_route-wp' ),
        )
    );

    // Switch default core markup for search form, comment form, and comments to output valid HTML5.
    add_theme_support(
        'html5',
        array(
            'search-form',
            'comment-form',
            'comment-list',
            'gallery',
            'caption',
            'style',
            'script',
        )
    );
}
add_action( 'after_setup_theme', 'wsmart_route_wp_setup' );

function wsmart_route_wp_scripts() {
    wp_enqueue_style( 'wsmart_route-wp-style', get_stylesheet_uri(), array(), '1.0.0' );
}
add_action( 'wp_enqueue_scripts', 'wsmart_route_wp_scripts' );
